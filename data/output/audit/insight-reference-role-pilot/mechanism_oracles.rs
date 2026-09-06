// SPDX-License-Identifier: GPL-2.0-or-later
// Independent arithmetic and finite permutation witnesses for reference-role review.

fn qualifying_triples(values: &[i64]) -> usize {
    let mut qualifying = 0;
    for first in 0..values.len() {
        for second in first + 1..values.len() {
            for third in second + 1..values.len() {
                let mut distances = [
                    (values[first] - values[second]).abs(),
                    (values[first] - values[third]).abs(),
                    (values[second] - values[third]).abs(),
                ];
                distances.sort();
                if distances[2] == 0 || 20 * (distances[2] - distances[1]) < distances[2] {
                    qualifying += 1;
                }
            }
        }
    }
    qualifying
}

fn check_permutations(values: &mut [i64], offset: usize, expected: usize) -> usize {
    if offset == values.len() {
        assert_eq!(qualifying_triples(values), expected);
        return 1;
    }
    let mut permutations = 0;
    for selected in offset..values.len() {
        values.swap(offset, selected);
        permutations += check_permutations(values, offset + 1, expected);
        values.swap(offset, selected);
    }
    permutations
}

fn binary_adic_distance(first: i64, second: i64) -> f64 {
    let difference = first.abs_diff(second);
    if difference == 0 {
        0.0
    } else {
        2.0_f64.powi(-(difference.trailing_zeros() as i32))
    }
}

fn main() {
    // Exhaustive relabeling tests the null mechanism, rather than replaying sampled data.
    let mut fixture = [0, 1, 2, 20, 40, 80];
    let expected = qualifying_triples(&fixture);
    let permutations = check_permutations(&mut fixture, 0, expected);
    assert_eq!(permutations, 720);
    assert!(expected > 0 && expected < 20);
    let mut binary_adic_triples = 0;
    for first in 0..16 {
        for second in first + 1..16 {
            for third in second + 1..16 {
                let mut distances = [
                    binary_adic_distance(first, second),
                    binary_adic_distance(first, third),
                    binary_adic_distance(second, third),
                ];
                distances.sort_by(f64::total_cmp);
                assert_eq!(distances[1], distances[2]);
                binary_adic_triples += 1;
            }
        }
    }
    assert_eq!(binary_adic_triples, 560);

    let mut width = 10.0_f64;
    let mut halvings = 0;
    while width >= 1e-8 {
        width *= 0.5;
        halvings += 1;
    }
    assert_eq!(halvings, 30);
    let prefactor = 3.0 * 2.99792458e10 * (67.36 * 1e5 / 3.0857e24) * 0.0493 * 0.83
        / (8.0 * std::f64::consts::PI * 6.67430e-8 * 1.67262192e-24)
        / 3.0857e18;
    assert!((prefactor - 927.9365461216893).abs() < 1e-9);
    let source_electron_factor = 1.0 - 0.25 / 2.0;
    let reported_width = 0.0008;
    let fourier_bin_spacing = 1.0 / 5045.0;
    assert!(reported_width > fourier_bin_spacing);

    println!("schema_version = 1");
    println!(
        "scope = \"Independent finite and arithmetic witnesses; scientific producers were not replayed\""
    );
    println!("permutations_checked = {permutations}");
    println!("qualifying_triples_per_permutation = {expected}");
    println!("total_triples_per_permutation = 20");
    println!("binary_adic_triples_checked = {binary_adic_triples}");
    println!("initial_bisection_width = 10.0");
    println!("bisection_tolerance = 1e-8");
    println!("required_halvings = {halvings}");
    println!("code_constants_prefactor = {prefactor:.12}");
    println!("source_helium_factor = {source_electron_factor}");
    println!(
        "same_constants_with_source_helium = {:.12}",
        prefactor * source_electron_factor
    );
    println!("reported_fwhm = {reported_width}");
    println!("fourier_bin_spacing = {fourier_bin_spacing:.12}");
    println!("rank_samples_per_cycle_at_0_214 = {:.12}", 1.0 / 0.214);
}
