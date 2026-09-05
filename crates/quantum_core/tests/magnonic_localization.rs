use faer::{Mat, Side, c64};
use quantum_core::{
    magnonic_crystal::{
        InversionBreakingParams, MagnonicTBParams, build_domain_wall_supercell,
        build_magnonic_9band, point_defect_modes,
    },
    tight_binding::{BravaisLattice2D, Hopping, TightBindingModel},
};

const TOLERANCE: f64 = 1e-10;

fn patch(base: &TightBindingModel, radius: usize, defect: bool) -> (TightBindingModel, Vec<usize>) {
    let side = 2 * radius + 1;
    let mut orbitals = Vec::new();
    let mut cells = Vec::new();
    let mut mapping = vec![None; side * side * 9];
    for cell in 0..side * side {
        let row = cell / side;
        let column = cell % side;
        for local in 0..9 {
            if defect && row == radius && column == radius && local == 6 {
                continue;
            }
            let mut orbital = base.orbitals[local].clone();
            orbital.position = orbital.position
                + base.lattice.a1.scale(row as f64)
                + base.lattice.a2.scale(column as f64);
            mapping[cell * 9 + local] = Some(orbitals.len());
            orbitals.push(orbital);
            cells.push(cell);
        }
    }
    let mut hoppings = Vec::new();
    for cell in 0..side * side {
        for hopping in &base.hoppings {
            let target_row = (cell / side) as i32 + hopping.cell_offset[0];
            let target_column = (cell % side) as i32 + hopping.cell_offset[1];
            if !(0..side as i32).contains(&target_row) || !(0..side as i32).contains(&target_column)
            {
                continue;
            }
            let target = target_row as usize * side + target_column as usize;
            if let (Some(from), Some(to)) = (
                mapping[cell * 9 + hopping.from],
                mapping[target * 9 + hopping.to],
            ) {
                hoppings.push(Hopping {
                    from,
                    to,
                    cell_offset: [0, 0],
                    amplitude: hopping.amplitude,
                });
            }
        }
    }
    (
        TightBindingModel {
            lattice: BravaisLattice2D::from_direct(
                base.lattice.a1.scale(side as f64),
                base.lattice.a2.scale(side as f64),
            ),
            orbitals,
            hoppings,
        },
        cells,
    )
}

fn sampled_gap(base: &TightBindingModel) -> (f64, f64) {
    let mut minima = [f64::INFINITY; 9];
    let mut maxima = [f64::NEG_INFINITY; 9];
    for row in 0..30 {
        for column in 0..30 {
            let momentum = base.lattice.b1.scale(row as f64 / 30.0)
                + base.lattice.b2.scale(column as f64 / 30.0);
            for (band, energy) in base
                .band_energies(momentum.x, momentum.y)
                .iter()
                .enumerate()
            {
                minima[band] = minima[band].min(*energy);
                maxima[band] = maxima[band].max(*energy);
            }
        }
    }
    (0..8)
        .map(|band| (maxima[band], minima[band + 1]))
        .filter(|(low, high)| high > low)
        .max_by(|left, right| (left.1 - left.0).total_cmp(&(right.1 - right.0)))
        .expect("sampled reference model must have a positive gap")
}

fn spectrum(hamiltonian: &Mat<c64>) -> (Vec<f64>, Mat<c64>) {
    let decomposition = hamiltonian.self_adjoint_eigen(Side::Lower).unwrap();
    let size = hamiltonian.nrows();
    let mut order: Vec<_> = (0..size).collect();
    order.sort_by(|left, right| {
        decomposition.S().column_vector()[*left]
            .re
            .total_cmp(&decomposition.S().column_vector()[*right].re)
    });
    let energies: Vec<_> = order
        .iter()
        .map(|index| decomposition.S().column_vector()[*index].re)
        .collect();
    let vectors = Mat::from_fn(size, size, |row, column| {
        decomposition.U()[(row, order[column])]
    });
    let product = hamiltonian * &vectors;
    let matrix_norm = (0..size)
        .flat_map(|row| (0..size).map(move |column| hamiltonian[(row, column)].norm_sqr()))
        .sum::<f64>()
        .sqrt();
    for column in 0..size {
        let norm = (0..size)
            .map(|row| vectors[(row, column)].norm_sqr())
            .sum::<f64>();
        let residual = (0..size)
            .map(|row| {
                (product[(row, column)] - vectors[(row, column)] * energies[column]).norm_sqr()
            })
            .sum::<f64>()
            .sqrt();
        assert!((norm - 1.0).abs() < TOLERANCE);
        assert!(residual / matrix_norm.max(1.0) < TOLERANCE);
    }
    (energies, vectors)
}

fn observations(
    label: &str,
    energies: &[f64],
    vectors: &Mat<c64>,
    first_mask: &[bool],
    second_mask: &[bool],
) {
    let mut first_weights = Vec::new();
    let mut second_weights = Vec::new();
    for (column, energy) in energies.iter().enumerate() {
        let mut inverse_participation = 0.0;
        let mut first = 0.0;
        let mut second = 0.0;
        for row in 0..vectors.nrows() {
            let probability = vectors[(row, column)].norm_sqr();
            inverse_participation += probability * probability;
            if first_mask[row] {
                first += probability;
            }
            if second_mask[row] {
                second += probability;
            }
        }
        assert!(inverse_participation >= 1.0 / vectors.nrows() as f64 - TOLERANCE);
        assert!(inverse_participation <= 1.0 + TOLERANCE);
        assert!(first + second <= 1.0 + TOLERANCE);
        first_weights.push(first);
        second_weights.push(second);
        println!(
            "MODE,{label},{column},{energy:.17},{inverse_participation:.17},{:.17},{first:.17},{second:.17},{:.17}",
            1.0 / inverse_participation,
            1.0 - first - second
        );
    }
    let mut start = 0;
    while start < energies.len() {
        let mut end = start + 1;
        while end < energies.len() && energies[end] - energies[start] <= TOLERANCE {
            end += 1;
        }
        let first_trace = first_weights[start..end].iter().sum::<f64>();
        let second_trace = second_weights[start..end].iter().sum::<f64>();
        println!(
            "GROUP,{label},{start},{end},{:.17},{first_trace:.17},{second_trace:.17}",
            energies[start]
        );
        start = end;
    }
}

#[test]
fn paired_open_patches_preserve_selector_and_measure_participation() {
    let parameters = MagnonicTBParams::kaman_default();
    let inversion = InversionBreakingParams::kaman_default();
    let base = build_magnonic_9band(&parameters, &inversion, 100.0);
    let (gap_low, gap_high) = sampled_gap(&base);
    println!("GAP,{gap_low:.17},{gap_high:.17}");
    println!(
        "MODE_COLUMNS,label,index,energy,ipr,participation_number,center_or_wall_weight,edge_or_seam_weight,remainder"
    );
    for radius in [2, 3, 4] {
        let side = 2 * radius + 1;
        for defect in [false, true] {
            let (model, cells) = patch(&base, radius, defect);
            let (energies, vectors) = spectrum(&model.hamiltonian_at_k(0.0, 0.0));
            let selected: Vec<_> = energies
                .iter()
                .copied()
                .filter(|energy| *energy > gap_low && *energy < gap_high)
                .collect();
            if defect {
                let producer = point_defect_modes(&parameters, &inversion, radius, 100.0);
                assert_eq!(selected.len(), producer.len());
                for (independent, actual) in selected.iter().zip(producer) {
                    assert!((independent - actual).abs() < TOLERANCE);
                }
            }
            let center: Vec<_> = cells
                .iter()
                .map(|cell| *cell == radius * side + radius)
                .collect();
            let edge: Vec<_> = cells
                .iter()
                .map(|cell| {
                    cell / side == 0
                        || cell / side == side - 1
                        || cell % side == 0
                        || cell % side == side - 1
                })
                .collect();
            let label = format!(
                "radius{radius}_{}",
                if defect { "defect" } else { "pristine" }
            );
            println!(
                "COUNT,{label},{},{},{}",
                energies.len(),
                selected.len(),
                side * side
            );
            observations(&label, &energies, &vectors, &center, &edge);
        }
    }
}

#[test]
fn periodic_wall_geometry_and_both_wall_weights() {
    let parameters = MagnonicTBParams::kaman_default();
    let inversion = InversionBreakingParams::kaman_default();
    let model = build_domain_wall_supercell(&parameters, &inversion, 20, 100.0);
    let signs: Vec<_> = (0..20)
        .map(|cell| (model.orbitals[cell * 9].on_site_energy - parameters.eps_s).signum())
        .collect();
    let transitions = (0..20)
        .filter(|cell| signs[*cell] != signs[(*cell + 1) % 20])
        .count();
    let seam_hoppings = model
        .hoppings
        .iter()
        .filter(|hopping| hopping.cell_offset[1] != 0 && hopping.amplitude.norm_sqr() > 0.0)
        .count();
    assert_eq!(transitions, 2);
    assert!(seam_hoppings > 0);
    let momentum = model.lattice.b1.scale(0.30);
    let (energies, vectors) = spectrum(&model.hamiltonian_at_k(momentum.x, momentum.y));
    let central: Vec<_> = (0..180)
        .map(|orbital| [9, 10].contains(&(orbital / 9)))
        .collect();
    let seam: Vec<_> = (0..180)
        .map(|orbital| [19, 0].contains(&(orbital / 9)))
        .collect();
    println!("WALLS,transitions={transitions},seam_hoppings={seam_hoppings}");
    observations("walls20", &energies, &vectors, &central, &seam);
}
