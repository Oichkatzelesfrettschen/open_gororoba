use std::{f64::consts::PI, fmt::Write as _, fs, path::PathBuf};

use faer::{Side, c64};
use quantum_core::{
    magnonic_crystal::{InversionBreakingParams, MagnonicTBParams, build_magnonic_9band},
    tight_binding::{
        BravaisLattice2D, Hopping, OrbitalSite, TightBindingModel, TopologyAdmission, Vec2,
        checked_subspace_topology,
    },
};

const SOURCE_FIGURE_TOLERANCE_GHZ: f64 = 0.03;

// Kaman et al., Figure 5(c,d): honeycomb s,px,py and bond-midpoint kagome s.
// Distance selection and directional projections independently determine every bond.
fn geometric_source_model(
    parameters: &MagnonicTBParams,
    inversion: &InversionBreakingParams,
    rectangular: bool,
) -> TightBindingModel {
    let root_three = 3.0_f64.sqrt();
    let primitive = BravaisLattice2D::hexagonal(1.0);
    let lattice = if rectangular {
        BravaisLattice2D::from_direct(Vec2::new(1.0, 0.0), Vec2::new(0.0, root_three))
    } else {
        primitive.clone()
    };
    let mut orbitals = Vec::new();
    for copy in 0..if rectangular { 2 } else { 1 } {
        let displacement = primitive.a2.scale(copy as f64);
        for (position, label, energy) in [
            (
                Vec2::zero(),
                "s_A",
                parameters.eps_s + inversion.delta_eps_s,
            ),
            (
                Vec2::zero(),
                "px_A",
                parameters.eps_p + inversion.delta_eps_p,
            ),
            (
                Vec2::zero(),
                "py_A",
                parameters.eps_p + inversion.delta_eps_p,
            ),
            (
                Vec2::new(0.0, 1.0 / root_three),
                "s_B",
                parameters.eps_s - inversion.delta_eps_s,
            ),
            (
                Vec2::new(0.0, 1.0 / root_three),
                "px_B",
                parameters.eps_p - inversion.delta_eps_p,
            ),
            (
                Vec2::new(0.0, 1.0 / root_three),
                "py_B",
                parameters.eps_p - inversion.delta_eps_p,
            ),
            (Vec2::new(0.0, 0.5 / root_three), "s_K1", parameters.eps_k),
            (
                Vec2::new(-0.25, -0.25 / root_three),
                "s_K2",
                parameters.eps_k,
            ),
            (
                Vec2::new(0.25, -0.25 / root_three),
                "s_K3",
                parameters.eps_k,
            ),
        ] {
            orbitals.push(OrbitalSite {
                position: position + displacement,
                label: format!("{label}_{copy}"),
                on_site_energy: energy,
            });
        }
    }
    let distance = 0.5 / root_three;
    let mut hoppings = Vec::new();
    for (from, origin) in orbitals
        .iter()
        .enumerate()
        .filter(|(index, _)| index % 9 < 6)
    {
        for (to, target) in orbitals
            .iter()
            .enumerate()
            .filter(|(index, _)| index % 9 >= 6)
        {
            for horizontal in -1..=1 {
                for vertical in -1..=1 {
                    let direction = target.position
                        + lattice.a1.scale(horizontal as f64)
                        + lattice.a2.scale(vertical as f64)
                        - origin.position;
                    if (direction.dot(direction) - distance * distance).abs() > 1e-12 {
                        continue;
                    }
                    let amplitude = match from % 3 {
                        0 => parameters.t_sk,
                        1 => parameters.t_pk * direction.x / distance,
                        _ => parameters.t_pk * direction.y / distance,
                    };
                    hoppings.push(Hopping {
                        from,
                        to,
                        cell_offset: [horizontal, vertical],
                        amplitude: c64::new(amplitude, 0.0),
                    });
                }
            }
        }
    }
    TightBindingModel {
        lattice,
        orbitals,
        hoppings,
    }
}

fn evidence_root() -> PathBuf {
    if let Some(root) = std::env::var_os("MAGNONIC_SOURCE_EVIDENCE_DIR") {
        return root.into();
    }
    let mut root = std::env::current_dir().expect("working directory");
    while !root
        .join("registry/canonical/control_plane.sqlite3")
        .is_file()
    {
        assert!(
            root.pop(),
            "run inside the repository or set MAGNONIC_SOURCE_EVIDENCE_DIR"
        );
    }
    root.join("data/output/audit/magnonic-source-projection")
}

fn retain(name: &str, text: &str) {
    if std::env::var_os("MAGNONIC_SOURCE_WRITE_RESULTS").is_some() {
        fs::write(evidence_root().join(name), text).expect("retain result");
    }
    println!("{text}");
}

#[test]
fn source_geometry_and_rectangular_folding_conform() {
    let mut report = String::from("schema_version = 1\nenergy_unit = 'GHz'\n");
    for (label, parameters, inversion) in [
        (
            "table_i",
            MagnonicTBParams::kaman_table_i(),
            InversionBreakingParams::none(),
        ),
        (
            "table_ii",
            MagnonicTBParams::kaman_table_ii(),
            InversionBreakingParams::kaman_table_ii(),
        ),
    ] {
        let implementation = build_magnonic_9band(&parameters, &inversion, 1.0);
        let geometric = geometric_source_model(&parameters, &inversion, false);
        let rectangular = geometric_source_model(&parameters, &inversion, true);
        assert_eq!(geometric.hoppings.len(), 18);
        assert_eq!(rectangular.hoppings.len(), 36);
        for grid in [12, 24, 48] {
            let mut maximum_matrix_error = 0.0_f64;
            let mut maximum_folding_error = 0.0_f64;
            for horizontal in 0..grid {
                for vertical in 0..grid {
                    let momentum = Vec2::new(
                        -PI + 2.0 * PI * horizontal as f64 / grid as f64,
                        -PI / 3.0_f64.sqrt()
                            + 2.0 * PI * vertical as f64 / (grid as f64 * 3.0_f64.sqrt()),
                    );
                    let actual = implementation.hamiltonian_at_k(momentum.x, momentum.y);
                    let expected = geometric.hamiltonian_at_k(momentum.x, momentum.y);
                    for row in 0..9 {
                        for column in 0..9 {
                            maximum_matrix_error = maximum_matrix_error
                                .max((actual[(row, column)] - expected[(row, column)]).norm());
                        }
                    }
                    let mut folded = implementation.band_energies(momentum.x, momentum.y);
                    folded.extend(
                        implementation
                            .band_energies(momentum.x, momentum.y + 2.0 * PI / 3.0_f64.sqrt()),
                    );
                    folded.sort_by(f64::total_cmp);
                    let direct = rectangular.band_energies(momentum.x, momentum.y);
                    for (expected, actual) in folded.iter().zip(direct) {
                        maximum_folding_error =
                            maximum_folding_error.max((expected - actual).abs());
                    }
                }
            }
            writeln!(report, "\n[[comparison]]\nparameters = '{label}'\ngrid = {grid}\nmaximum_matrix_error_ghz = {maximum_matrix_error:.16e}\nmaximum_rectangular_folding_error_ghz = {maximum_folding_error:.16e}").unwrap();
            assert!(maximum_matrix_error < 1e-11);
            assert!(maximum_folding_error < 1e-11);
        }
    }
    retain("geometry-folding-results.toml", &report);
}

fn read_ppm(name: &str) -> (usize, usize, Vec<u8>) {
    let bytes = fs::read(evidence_root().join(name)).expect("source figure PPM");
    let mut position = 0;
    let mut tokens = Vec::new();
    while tokens.len() < 4 {
        while bytes[position].is_ascii_whitespace() {
            position += 1;
        }
        if bytes[position] == b'#' {
            while bytes[position] != b'\n' {
                position += 1;
            }
            continue;
        }
        let start = position;
        while !bytes[position].is_ascii_whitespace() {
            position += 1;
        }
        tokens.push(
            std::str::from_utf8(&bytes[start..position])
                .unwrap()
                .to_string(),
        );
    }
    assert_eq!(tokens[0], "P6");
    assert_eq!(tokens[3], "255");
    position += 1;
    let width = tokens[1].parse::<usize>().unwrap();
    let height = tokens[2].parse::<usize>().unwrap();
    assert_eq!(bytes.len() - position, width * height * 3);
    (width, height, bytes[position..].to_vec())
}

fn directed_distance(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .map(|value| {
            right
                .iter()
                .map(|other| (value - other).abs())
                .fold(f64::INFINITY, f64::min)
        })
        .fold(0.0, f64::max)
}

#[test]
fn source_projected_spectral_support_discriminates_parameter_identity() {
    let (width, height, pixels) = read_ppm("source-figure5.ppm");
    assert_eq!((width, height), (866, 971));
    let mut report = format!(
        "schema_version = 1\nsource = 'Kaman Figure 5b'\nfigure_gate_ghz = {SOURCE_FIGURE_TOLERANCE_GHZ}\nplot_left_pixel = 395\nplot_right_pixel = 634\nplot_top_pixel = 74\nplot_bottom_pixel = 726\nminimum_frequency_ghz = 1.0\nmaximum_frequency_ghz = 4.5\n"
    );
    let mut csv = String::from(
        "parameters,ky_samples,pixel_threshold,pixel_x,kx_per_um,source_to_model_ghz,model_to_source_ghz\n",
    );
    for (label, parameters) in [
        ("table_i", MagnonicTBParams::kaman_table_i()),
        ("repository_reference", MagnonicTBParams::kaman_default()),
    ] {
        let model = geometric_source_model(&parameters, &InversionBreakingParams::none(), true);
        for samples in [96, 192] {
            for threshold in [220_u8, 240] {
                let mut maximum_distance = 0.0_f64;
                for pixel_x in (404..=624).step_by(5) {
                    let kx = -PI + 2.0 * PI * (pixel_x as f64 - 395.0) / 239.0;
                    let frequencies: Vec<f64> = (90..715)
                        .filter(|pixel_y| {
                            let offset = (pixel_y * width + pixel_x) * 3;
                            pixels[offset..offset + 3]
                                .iter()
                                .all(|channel| *channel < threshold)
                        })
                        .map(|pixel_y| 4.5 - 3.5 * (pixel_y as f64 - 74.0) / 652.0)
                        .collect();
                    assert!(!frequencies.is_empty());
                    let mut predicted = Vec::new();
                    for vertical in 0..samples {
                        let ky = -PI / 3.0_f64.sqrt()
                            + 2.0 * PI * vertical as f64 / (samples as f64 * 3.0_f64.sqrt());
                        predicted.extend(
                            model
                                .band_energies(kx, ky)
                                .into_iter()
                                .filter(|frequency| (1.06..4.40).contains(frequency)),
                        );
                    }
                    let source_to_model = directed_distance(&frequencies, &predicted);
                    let model_to_source = directed_distance(&predicted, &frequencies);
                    maximum_distance = maximum_distance.max(source_to_model).max(model_to_source);
                    writeln!(csv, "{label},{samples},{threshold},{pixel_x},{:.12},{source_to_model:.12},{model_to_source:.12}", kx/0.333).unwrap();
                }
                writeln!(report, "\n[[projection]]\nparameters = '{label}'\nky_samples = {samples}\npixel_threshold = {threshold}\ncolumn_count = 45\nmaximum_two_way_support_distance_ghz = {maximum_distance:.16e}\npasses_figure_gate = {}", maximum_distance <= SOURCE_FIGURE_TOLERANCE_GHZ).unwrap();
                if label == "repository_reference" {
                    assert!(maximum_distance > SOURCE_FIGURE_TOLERANCE_GHZ);
                } else {
                    assert!(maximum_distance <= SOURCE_FIGURE_TOLERANCE_GHZ);
                }
            }
        }
    }
    if std::env::var_os("MAGNONIC_SOURCE_WRITE_RESULTS").is_some() {
        fs::write(evidence_root().join("projection-columns.csv"), csv).unwrap();
    }
    retain("projection-results.toml", &report);
}

#[test]
fn source_subspace_topology_grid_refinement_retains_admission_failures() {
    let mut report = String::from(
        "schema_version = 1\nscope = 'Finite reciprocal-grid external gaps and determinant-link topology; grid refinement does not prove a global gap.'\n",
    );
    for (label, parameters, inversion) in [
        (
            "table_i",
            MagnonicTBParams::kaman_table_i(),
            InversionBreakingParams::none(),
        ),
        (
            "table_ii",
            MagnonicTBParams::kaman_table_ii(),
            InversionBreakingParams::kaman_table_ii(),
        ),
        (
            "table_ii_reversed",
            MagnonicTBParams::kaman_table_ii(),
            InversionBreakingParams {
                delta_eps_s: 0.12,
                delta_eps_p: 0.35,
            },
        ),
    ] {
        let model = geometric_source_model(&parameters, &inversion, false);
        for grid in [12, 24, 48, 96] {
            for bands in [
                0..1,
                1..2,
                2..3,
                3..4,
                4..5,
                5..6,
                6..7,
                7..8,
                8..9,
                0..2,
                2..5,
                5..9,
                2..4,
                2..6,
                4..6,
                6..9,
                7..9,
                2..9,
                0..9,
            ] {
                writeln!(report, "\n[[topology]]\nparameters = '{label}'\ngrid = {grid}\nbands_start = {}\nbands_end_exclusive = {}", bands.start, bands.end).unwrap();
                match checked_subspace_topology(
                    &model,
                    bands.clone(),
                    grid,
                    TopologyAdmission::default(),
                ) {
                    Ok(topology) => {
                        let chern = topology.chern_number();
                        writeln!(report, "admitted = true\nchern = {chern:.16e}\nminimum_link_determinant = {:.16e}", topology.minimum_link_determinant).unwrap();
                        if let Some(gap) = topology.minimum_sampled_gap {
                            writeln!(report, "minimum_sampled_external_gap_ghz = {gap:.16e}")
                                .unwrap();
                        }
                        assert!(
                            chern.abs() < 1e-8,
                            "real hopping model respects spinless time reversal"
                        );
                    }
                    Err(error) => {
                        writeln!(report, "admitted = false\nerror = '{error}'").unwrap();
                    }
                }
            }
        }
    }
    retain("topology-refinement-results.toml", &report);
}

#[test]
fn source_inversion_broken_projected_support_retains_rendered_scope() {
    let (width, height, pixels) = read_ppm("source-figure6.ppm");
    assert_eq!((width, height), (2114, 1433));
    let mut report = format!(
        "schema_version = 1\nsource = 'Kaman Figure 6a model panel'\nfigure_gate_ghz = {SOURCE_FIGURE_TOLERANCE_GHZ}\nplot_left_pixel = 693\nplot_right_pixel = 1153\nplot_top_pixel = 128\nplot_bottom_pixel = 1283\nminimum_frequency_ghz = 0.5\nmaximum_frequency_ghz = 3.0\nfrequency_windows_ghz = [[1.23,1.445],[1.56,1.99],[2.055,2.40],[2.80,2.93]]\nscope = 'Rendered lower bands only; frequency masks exclude inset and gap-label text, grayscale mask excludes cyan shading and purple valley markers.'\n"
    );
    let admitted_frequency = |frequency: f64| {
        [(1.23, 1.445), (1.56, 1.99), (2.055, 2.40), (2.80, 2.93)]
            .iter()
            .any(|(lower, upper)| (*lower..=*upper).contains(&frequency))
    };
    for (label, parameters, inversion) in [
        (
            "table_ii",
            MagnonicTBParams::kaman_table_ii(),
            InversionBreakingParams::kaman_table_ii(),
        ),
        (
            "repository_reference",
            MagnonicTBParams::kaman_default(),
            InversionBreakingParams::kaman_default(),
        ),
    ] {
        let model = geometric_source_model(&parameters, &inversion, true);
        for samples in [96, 192] {
            for threshold in [220_u8, 240] {
                let mut maximum_distance = 0.0_f64;
                let mut columns = 0;
                let mut witness = (0_usize, 0.0_f64, 0.0_f64, 0.0_f64, "");
                let mut excluded_inset_pixels = 0;
                let mut retained_source_pixels = 0;
                let mut retained_model_samples = 0;
                let mut occluded_model_samples = 0;
                for pixel_x in (706_usize..=1140).step_by(10) {
                    if pixel_x.abs_diff(768) < 6 || pixel_x.abs_diff(1076) < 6 {
                        continue;
                    }
                    let kx = -PI + 2.0 * PI * (pixel_x as f64 - 693.0) / 460.0;
                    let frequencies: Vec<f64> = (150..960)
                        .filter_map(|pixel_y| {
                            if (825..=1025).contains(&pixel_x) && (927..=1237).contains(&pixel_y) {
                                excluded_inset_pixels += 1;
                                return None;
                            }
                            let offset = (pixel_y * width + pixel_x) * 3;
                            let color = &pixels[offset..offset + 3];
                            let frequency = 3.0 - 2.5 * (pixel_y as f64 - 128.0) / 1155.0;
                            (admitted_frequency(frequency)
                                && color.iter().all(|channel| *channel < threshold)
                                && color.iter().max().unwrap() - color.iter().min().unwrap() < 10)
                                .then_some(frequency)
                        })
                        .collect();
                    assert!(!frequencies.is_empty());
                    retained_source_pixels += frequencies.len();
                    let mut predicted = Vec::new();
                    for vertical in 0..samples {
                        let ky = -PI / 3.0_f64.sqrt()
                            + 2.0 * PI * vertical as f64 / (samples as f64 * 3.0_f64.sqrt());
                        predicted.extend(model.band_energies(kx, ky).into_iter().filter(
                            |frequency| {
                                let pixel_y = 128.0 + (3.0 - frequency) * 1155.0 / 2.5;
                                let occluded = (825..=1025).contains(&pixel_x)
                                    && (927.0..=1237.0).contains(&pixel_y);
                                if occluded && admitted_frequency(*frequency) {
                                    occluded_model_samples += 1;
                                }
                                admitted_frequency(*frequency) && !occluded
                            },
                        ));
                    }
                    for (left, right, direction) in [
                        (&frequencies, &predicted, "source_to_model"),
                        (&predicted, &frequencies, "model_to_source"),
                    ] {
                        for frequency in left {
                            let nearest = right
                                .iter()
                                .min_by(|first, second| {
                                    (**first - frequency)
                                        .abs()
                                        .total_cmp(&(**second - frequency).abs())
                                })
                                .unwrap();
                            let difference = (frequency - nearest).abs();
                            if difference > maximum_distance {
                                maximum_distance = difference;
                                witness = (pixel_x, kx / 0.333, *frequency, *nearest, direction);
                            }
                        }
                    }
                    retained_model_samples += predicted.len();
                    columns += 1;
                }
                writeln!(report,"\n[[projection]]\nparameters = '{label}'\nky_samples = {samples}\npixel_threshold = {threshold}\ncolumn_count = {columns}\nmaximum_two_way_support_distance_ghz = {maximum_distance:.16e}\npasses_figure_gate = {}",maximum_distance<=SOURCE_FIGURE_TOLERANCE_GHZ).unwrap();
                writeln!(report,"witness_pixel_x = {}\nwitness_kx_per_um = {:.12}\nwitness_frequency_ghz = {:.12}\nwitness_nearest_frequency_ghz = {:.12}\nwitness_direction = '{}'",witness.0,witness.1,witness.2,witness.3,witness.4).unwrap();
                let target_frequency = if witness.4 == "source_to_model" {
                    witness.3
                } else {
                    witness.2
                };
                let mut matched_sample = (f64::INFINITY, 0.0, 0_usize);
                for vertical in 0..samples {
                    let ky = -PI / 3.0_f64.sqrt()
                        + 2.0 * PI * vertical as f64 / (samples as f64 * 3.0_f64.sqrt());
                    for (band, frequency) in model
                        .band_energies(witness.1 * 0.333, ky)
                        .into_iter()
                        .enumerate()
                    {
                        let difference = (frequency - target_frequency).abs();
                        if difference < matched_sample.0 {
                            matched_sample = (difference, ky / 0.333, band);
                        }
                    }
                }
                writeln!(report,"witness_model_ky_per_um = {:.12}\nwitness_rectangular_band_zero_based = {}\nwitness_model_frequency_match_error_ghz = {:.16e}",matched_sample.1,matched_sample.2,matched_sample.0).unwrap();
                writeln!(report,"inset_exclusion_rectangle_pixels = [825,1025,927,1237]\nexcluded_inset_pixels_in_scanned_columns = {excluded_inset_pixels}\nretained_source_pixels = {retained_source_pixels}\nretained_model_frequency_samples = {retained_model_samples}").unwrap();
                writeln!(
                    report,
                    "occluded_model_frequency_samples = {occluded_model_samples}"
                )
                .unwrap();
                if label == "repository_reference" {
                    assert!(maximum_distance > SOURCE_FIGURE_TOLERANCE_GHZ);
                } else {
                    assert!(maximum_distance <= SOURCE_FIGURE_TOLERANCE_GHZ);
                }
            }
        }
    }
    retain("inversion-projection-results.toml", &report);
}

// Kaman et al., Supplemental V uses Bloch orbitals in PythTB. The position
// phases convert the periodic-cell eigenvectors to that embedded convention.
fn local_valley_flux(
    model: &TightBindingModel,
    grid: usize,
    valley_sign: f64,
    embedded: bool,
) -> ([f64; 4], f64, f64) {
    let center = Vec2::new(valley_sign * 4.0 * PI / 3.0, 0.0);
    let radius = 0.1 * 4.0 * PI / 3.0;
    let selected = [0, 1, 3, 4];
    let mut minimum_gap = f64::INFINITY;
    let mut frames = vec![vec![vec![[c64::new(0.0, 0.0); 9]; 4]; grid + 1]; grid + 1];
    for (horizontal, row) in frames.iter_mut().enumerate() {
        for (vertical, frames) in row.iter_mut().enumerate() {
            let momentum = Vec2::new(
                center.x - radius + 2.0 * radius * horizontal as f64 / grid as f64,
                center.y - radius + 2.0 * radius * vertical as f64 / grid as f64,
            );
            let hamiltonian = model.hamiltonian_at_k(momentum.x, momentum.y);
            let eigen = hamiltonian.self_adjoint_eigen(Side::Lower).unwrap();
            let values = eigen.S().column_vector();
            let mut order: Vec<usize> = (0..9).collect();
            order.sort_by(|left, right| values[*left].re.total_cmp(&values[*right].re));
            for (slot, band) in selected.into_iter().enumerate() {
                for boundary in [band, band + 1] {
                    if boundary > 0 && boundary < 9 {
                        minimum_gap = minimum_gap
                            .min(values[order[boundary]].re - values[order[boundary - 1]].re);
                    }
                }
                for (orbital, component) in frames[slot].iter_mut().enumerate() {
                    let angle = if embedded {
                        -momentum.dot(model.orbitals[orbital].position)
                    } else {
                        0.0
                    };
                    *component =
                        c64::new(angle.cos(), angle.sin()) * eigen.U()[(orbital, order[band])];
                }
            }
        }
    }
    assert!(
        minimum_gap > 1e-10,
        "individual bands must be isolated throughout sampled valley patch"
    );
    let mut minimum_overlap = f64::INFINITY;
    let mut flux = [0.0; 4];
    for horizontal in 0..grid {
        for vertical in 0..grid {
            let corners = [
                (horizontal, vertical),
                (horizontal + 1, vertical),
                (horizontal + 1, vertical + 1),
                (horizontal, vertical + 1),
            ];
            for slot in 0..4 {
                let mut product = c64::new(1.0, 0.0);
                for edge in 0..4 {
                    let (left_x, left_y) = corners[edge];
                    let (right_x, right_y) = corners[(edge + 1) % 4];
                    let overlap: c64 = frames[left_x][left_y][slot]
                        .iter()
                        .zip(&frames[right_x][right_y][slot])
                        .map(|(left, right)| c64::new(left.re, -left.im) * right)
                        .sum();
                    minimum_overlap = minimum_overlap.min(overlap.norm());
                    assert!(overlap.norm() > 1e-12);
                    product *= overlap / overlap.norm();
                }
                flux[slot] += product.im.atan2(product.re) / (2.0 * PI);
            }
        }
    }
    (flux, minimum_gap, minimum_overlap)
}

#[test]
fn embedded_valley_patch_flux_tests_source_sign_predictions() {
    let mut report = String::from(
        "schema_version = 1\nsource = 'Kaman Supplemental V, Berry curvature calculations in inversion-broken crystals'\nbands = [0, 1, 3, 4]\npatch = 'Cartesian square centered at (+/-4*pi/3,0), half width 0.1*4*pi/3, lattice constant 1'\nscope = 'Local finite-patch flux; individual-band admission applies to patch samples and does not imply global band isolation or quantization.'\n",
    );
    for scale in [0.2, 1.0] {
        for grid in [12, 24, 48, 96] {
            let mut results = Vec::new();
            for mass_sign in [1.0, -1.0] {
                let model = geometric_source_model(
                    &MagnonicTBParams::kaman_table_ii(),
                    &InversionBreakingParams {
                        delta_eps_s: -0.12 * scale * mass_sign,
                        delta_eps_p: -0.35 * scale * mass_sign,
                    },
                    false,
                );
                for valley_sign in [1.0, -1.0] {
                    for embedded in [true, false] {
                        let (flux, gap, overlap) =
                            local_valley_flux(&model, grid, valley_sign, embedded);
                        writeln!(report,"\n[[valley]]\nmass_scale = {scale}\nmass_sign = {mass_sign}\nvalley_sign = {valley_sign}\ngrid = {grid}\nembedded_orbital_gauge = {embedded}\nflux = {flux:?}\nminimum_sampled_patch_gap_ghz = {gap:.16e}\nminimum_overlap = {overlap:.16e}").unwrap();
                        if embedded {
                            results.push(flux);
                        }
                    }
                }
            }
            for ((original, opposite_valley), opposite_mass) in
                results[0].iter().zip(results[1]).zip(results[2])
            {
                assert!(
                    (original + opposite_valley).abs() < 1e-9,
                    "time reversal exchanges valleys"
                );
                assert!(
                    (original + opposite_mass).abs() < 1e-9,
                    "inversion exchanges mass signs"
                );
            }
            if scale < 0.5 {
                assert!(results[0][0] * results[0][1] < 0.0);
                assert!(results[0][2] * results[0][3] < 0.0);
                assert!(results[0][0] * results[0][2] < 0.0);
            }
        }
    }
    retain("valley-patch-results.toml", &report);
}
