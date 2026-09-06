use quantum_core::{
    Complex64,
    magnonic_crystal::{InversionBreakingParams, MagnonicTBParams, build_magnonic_9band},
    spectral_gap_certificate::{certify_global_gaps, validate_cover},
    tight_binding::{BravaisLattice2D, Hopping, OrbitalSite, TightBindingModel, Vec2},
};
use std::{fmt::Write as _, path::Path};

fn diagonal_model(gap: f64) -> TightBindingModel {
    TightBindingModel {
        lattice: BravaisLattice2D::square(1.0),
        orbitals: [0.0, gap]
            .into_iter()
            .enumerate()
            .map(|(index, on_site_energy)| OrbitalSite {
                position: Vec2::zero(),
                label: format!("orbital_{index}"),
                on_site_energy,
            })
            .collect(),
        hoppings: Vec::new(),
    }
}

#[test]
fn exact_gap_degeneracy_and_cover_mutations() {
    let certificate = certify_global_gaps(&diagonal_model(1.0), &[1], 2, 1e-10).unwrap();
    assert!(certificate.admitted);
    assert!(certificate.leaves[0].gap_lower > 0.999999999);
    assert!(!validate_cover(&[], 2));
    let mut duplicate = certificate.leaves.clone();
    duplicate.push(duplicate[0].clone());
    assert!(!validate_cover(&duplicate, 2));
    let mut overlap = certificate.leaves.clone();
    let mut child = overlap[0].clone();
    child.depth = 1;
    overlap.push(child);
    assert!(!validate_cover(&overlap, 2));
    let mut outside = certificate.leaves.clone();
    outside[0].coordinates = [1, 0];
    assert!(!validate_cover(&outside, 2));
    let degenerate = certify_global_gaps(&diagonal_model(0.0), &[1], 2, 1e-10).unwrap();
    assert!(!degenerate.admitted);
    assert_eq!(degenerate.leaves.len(), 16);
    assert!(validate_cover(&degenerate.leaves, 2));
    let mut omitted = degenerate.leaves.clone();
    omitted.pop();
    assert!(!validate_cover(&omitted, 2));
    // A translated diagonal hop closes the gap at a torus point.
    let mut closing = diagonal_model(1.0);
    closing.hoppings.push(Hopping {
        from: 1,
        to: 1,
        cell_offset: [1, 0],
        amplitude: Complex64::new(0.5, 0.0),
    });
    let closed = certify_global_gaps(&closing, &[1], 4, 1e-10).unwrap();
    assert!(!closed.admitted);
    assert!(closed.hopping_row_upper >= 1.0);
}

fn retain_model(model: &TightBindingModel, path: &Path) {
    let mut output =
        String::from("kind,index,from,to,offset_u,offset_v,real_bits,imaginary_bits\n");
    for (index, orbital) in model.orbitals.iter().enumerate() {
        writeln!(
            output,
            "onsite,{index},{index},{index},0,0,{:016x},0000000000000000",
            orbital.on_site_energy.to_bits()
        )
        .unwrap();
    }
    for (index, hop) in model.hoppings.iter().enumerate() {
        writeln!(
            output,
            "hopping,{index},{},{},{},{},{:016x},{:016x}",
            hop.from,
            hop.to,
            hop.cell_offset[0],
            hop.cell_offset[1],
            hop.amplitude.re.to_bits(),
            hop.amplitude.im.to_bits()
        )
        .unwrap();
    }
    std::fs::write(path, output).unwrap();
}

#[test]
#[ignore = "Explicit retained numerical campaign"]
fn retain_table_i_then_table_ii_global_certificate() {
    let destination = std::env::var("MAGNONIC_GAP_OUTPUT").expect("Retained output path required");
    let destination = Path::new(&destination);
    std::fs::create_dir_all(destination).unwrap();
    let cases = [
        (
            "table-i",
            MagnonicTBParams::kaman_table_i(),
            InversionBreakingParams::none(),
            vec![2, 6],
        ),
        (
            "table-ii",
            MagnonicTBParams::kaman_table_ii(),
            InversionBreakingParams::kaman_table_ii(),
            vec![1, 2, 4, 6, 7],
        ),
    ];
    let mut report = String::from(
        "schema_version = 1\ncoefficient_scope = \"exact stored dyadic values\"\nmaximum_depth = 10\ngap_gate_ghz = 1e-10\n",
    );
    for (name, parameters, inversion, boundaries) in cases {
        let model = build_magnonic_9band(&parameters, &inversion, 1.0);
        assert!(model.hoppings.iter().all(|hop| hop.amplitude.im == 0.0));
        retain_model(
            &model,
            &destination.join(format!("{name}-coefficient-bits.csv")),
        );
        let certificate = certify_global_gaps(&model, &boundaries, 10, 1e-10).unwrap();
        let mut cells = String::from(
            "depth,index_u,index_v,gap_lower_ghz,eigenvalue_radius_ghz,eta_upper,rho_upper_ghz,gap_variation_upper_ghz",
        );
        for band in 0..9 {
            write!(cells, ",eigenvalue_{band}_ghz").unwrap();
        }
        cells.push('\n');
        let mut production_difference: f64 = 0.0;
        let mut minimum_gap = f64::INFINITY;
        let mut maximum_radius: f64 = 0.0;
        let mut denominator_count = 0_u64;
        for cell in &certificate.leaves {
            write!(
                cells,
                "{},{},{},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e}",
                cell.depth,
                cell.coordinates[0],
                cell.coordinates[1],
                cell.gap_lower,
                cell.eigenvalue_radius,
                cell.orthogonality_upper,
                cell.residual_upper,
                cell.variation_upper
            )
            .unwrap();
            for value in &cell.center_eigenvalues {
                write!(cells, ",{value:.17e}").unwrap();
            }
            cells.push('\n');
            let denominator = f64::from(1_u32 << cell.depth);
            let horizontal = (f64::from(cell.coordinates[0]) + 0.5) / denominator;
            let vertical = (f64::from(cell.coordinates[1]) + 0.5) / denominator;
            let point = Vec2::new(
                horizontal * model.lattice.b1.x + vertical * model.lattice.b2.x,
                horizontal * model.lattice.b1.y + vertical * model.lattice.b2.y,
            );
            let energies = model.band_energies(point.x, point.y);
            for (production, certified) in energies.iter().zip(&cell.center_eigenvalues) {
                production_difference = production_difference.max((production - certified).abs());
            }
            minimum_gap = minimum_gap.min(cell.gap_lower);
            maximum_radius = maximum_radius.max(cell.eigenvalue_radius);
            denominator_count += 1_u64 << (2 * (10 - cell.depth));
        }
        std::fs::write(
            destination.join(format!("{name}-terminal-cells.csv")),
            cells,
        )
        .unwrap();
        let unresolved = certificate
            .leaves
            .iter()
            .filter(|cell| cell.gap_lower <= 1e-10)
            .count();
        writeln!(report,"\n[[model]]\nname = \"{name}\"\nboundaries = {boundaries:?}\nreal_hoppings_exact = true\nadmitted = {}\nterminal_cells = {}\nunresolved_cells = {unresolved}\ncovered_integer_area = {denominator_count}\nexpected_integer_area = 1048576\nhopping_row_upper = {:.17e}\nminimum_gap_lower_ghz = {minimum_gap:.17e}\nmaximum_eigenvalue_radius_ghz = {maximum_radius:.17e}\nmaximum_sampled_production_eigenvalue_difference_ghz = {production_difference:.17e}\nproduction_comparison_scope = \"Sampled Cartesian evaluator consistency; excluded from global certificate proof\"",certificate.admitted,certificate.leaves.len(),certificate.hopping_row_upper).unwrap();
        std::fs::write(destination.join("results.toml"), &report).unwrap();
        assert!(
            certificate.admitted,
            "{name}: {unresolved} unresolved cells"
        );
        assert!(production_difference < 1e-11);
    }
}
