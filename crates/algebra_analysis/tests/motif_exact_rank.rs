//! Exact finite adjacency witnesses avoid thresholded eigenspectrum rank claims.
//! Identical integer columns give a rational rank upper bound; a modular rank
//! meeting that bound proves equality over the rationals without approximation.
use algebra_analysis::boxkites::motif_components_for_cross_assessors;
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::Write as _,
};

const PRIME: u64 = 1_000_000_007;

fn power_mod(mut base: u64, mut exponent: u64) -> u64 {
    let mut result = 1;
    while exponent > 0 {
        if exponent % 2 == 1 {
            result = result * base % PRIME;
        }
        base = base * base % PRIME;
        exponent /= 2;
    }
    result
}

fn modular_rank(mut matrix: Vec<Vec<u64>>) -> usize {
    let mut rank = 0;
    for column in 0..matrix[0].len() {
        let Some(pivot) = (rank..matrix.len()).find(|&row| matrix[row][column] != 0) else {
            continue;
        };
        matrix.swap(rank, pivot);
        let inverse = power_mod(matrix[rank][column], PRIME - 2);
        for row in rank + 1..matrix.len() {
            let factor = matrix[row][column] * inverse % PRIME;
            for index in column..matrix[0].len() {
                matrix[row][index] =
                    (matrix[row][index] + PRIME - factor * matrix[rank][index] % PRIME) % PRIME;
            }
        }
        rank += 1;
        if rank == matrix.len() {
            break;
        }
    }
    rank
}

fn verify_integer_laplacian_eigenvector(matrix: &[Vec<u64>], vector: &[i64], eigenvalue: i64) {
    for (row_index, row) in matrix.iter().enumerate() {
        let degree = row.iter().sum::<u64>() as i64;
        let actual = degree * vector[row_index]
            - row
                .iter()
                .zip(vector)
                .map(|(entry, value)| *entry as i64 * value)
                .sum::<i64>();
        assert_eq!(actual, eigenvalue * vector[row_index]);
    }
}

fn complement_components(matrix: &[Vec<u64>]) -> usize {
    let mut visited = vec![false; matrix.len()];
    let mut count = 0;
    for start in 0..matrix.len() {
        if visited[start] {
            continue;
        }
        count += 1;
        let mut frontier = vec![start];
        visited[start] = true;
        while let Some(current) = frontier.pop() {
            for neighbor in 0..matrix.len() {
                if current != neighbor && matrix[current][neighbor] == 0 && !visited[neighbor] {
                    visited[neighbor] = true;
                    frontier.push(neighbor);
                }
            }
        }
    }
    count
}

#[test]
fn every_enumerated_motif_has_certified_rational_rank() {
    let mut report = String::from(
        "schema_version = 1\ndimensions = [16, 32, 64, 128, 256]\nrank_field = \"Q\"\nrank_lower_bound_field = \"F_1000000007\"\nupper_bound = \"Number of distinct integer adjacency columns\"\nuniversal_extension = \"unproved\"\n",
    );
    for dimension in [16, 32, 64, 128, 256] {
        let components = motif_components_for_cross_assessors(dimension);
        assert!(!components.is_empty());
        let mut dimension_minimum_degree = usize::MAX;
        let mut dimension_maximum_degree = 0;
        let mut cocktail_components = 0;
        let mut degree_profiles = BTreeSet::new();
        for (component_index, component) in components.iter().enumerate() {
            let nodes: Vec<_> = component.nodes.iter().copied().collect();
            let indices: BTreeMap<_, _> = nodes
                .iter()
                .enumerate()
                .map(|(index, node)| (*node, index))
                .collect();
            let vertex_count = nodes.len();
            assert_eq!(vertex_count, dimension / 2 - 2);
            let mut matrix = vec![vec![0u64; vertex_count]; vertex_count];
            for (left, right) in &component.edges {
                let row = indices[left];
                let column = indices[right];
                assert_ne!(row, column);
                assert_eq!(matrix[row][column], 0);
                matrix[row][column] = 1;
                matrix[column][row] = 1;
            }
            let mut distinct_columns: BTreeMap<Vec<u64>, Vec<usize>> = BTreeMap::new();
            for column in 0..vertex_count {
                let values: Vec<_> = matrix.iter().map(|row| row[column]).collect();
                distinct_columns.entry(values).or_default().push(column);
            }
            let upper_bound = distinct_columns.len();
            let lower_bound = modular_rank(matrix.clone());
            assert_eq!(
                lower_bound, upper_bound,
                "D={dimension}, component={component_index}: rank certificate incomplete"
            );
            assert_eq!(
                2 * (vertex_count - lower_bound),
                vertex_count,
                "D={dimension}, component={component_index}: nullity differs from half"
            );
            let mut degrees: Vec<_> = matrix.iter().map(|row| row.iter().sum::<u64>()).collect();
            degrees.sort_unstable();
            dimension_minimum_degree = dimension_minimum_degree.min(degrees[0] as usize);
            dimension_maximum_degree =
                dimension_maximum_degree.max(*degrees.last().unwrap() as usize);
            degree_profiles.insert(degrees.clone());
            let pairs: Vec<_> = distinct_columns.values().cloned().collect();
            let cocktail = pairs.iter().all(|pair| pair.len() == 2)
                && matrix.iter().enumerate().all(|(row, values)| {
                    values.iter().enumerate().all(|(column, value)| {
                        let same_part = pairs
                            .iter()
                            .any(|pair| pair.contains(&row) && pair.contains(&column));
                        *value == u64::from(!same_part)
                    })
                });
            cocktail_components += usize::from(cocktail);
            let complement_component_count = complement_components(&matrix);
            for pair in &pairs {
                assert_eq!(pair.len(), 2);
                let mut vector = vec![0; vertex_count];
                vector[pair[0]] = 1;
                vector[pair[1]] = -1;
                let degree = matrix[pair[0]].iter().sum::<u64>() as i64;
                verify_integer_laplacian_eigenvector(&matrix, &vector, degree);
            }
            if cocktail {
                verify_integer_laplacian_eigenvector(&matrix, &vec![1; vertex_count], 0);
                for pair in pairs.iter().skip(1) {
                    let mut vector = vec![0; vertex_count];
                    for vertex in pair {
                        vector[*vertex] = 1;
                    }
                    for vertex in &pairs[0] {
                        vector[*vertex] = -1;
                    }
                    verify_integer_laplacian_eigenvector(&matrix, &vector, vertex_count as i64);
                }
            }

            let row_bits: Vec<String> = matrix
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|value| if *value == 0 { '0' } else { '1' })
                        .collect()
                })
                .collect();
            writeln!(report, "\n[[component]]\ndimension = {dimension}\ncomponent = {component_index}\nvertices = {vertex_count}\nedges = {}\ndegree_minimum = {}\ndegree_maximum = {}\nrank_lower_bound_mod_prime = {lower_bound}\nrank_upper_bound_distinct_columns = {upper_bound}\nnullity = {}\ncocktail_party = {cocktail}\ncomplement_components = {complement_component_count}\nnodes = {nodes:?}\ncolumn_equivalence_classes = {pairs:?}\nadjacency_rows = {row_bits:?}", component.edges.len(), degrees[0], degrees.last().unwrap(), vertex_count - lower_bound).unwrap();
        }
        writeln!(report, "\n[[dimension_summary]]\ndimension = {dimension}\ncomponents = {}\nminimum_degree = {dimension_minimum_degree}\nmaximum_degree = {dimension_maximum_degree}\ndistinct_degree_profiles = {}\ncocktail_party_components = {cocktail_components}", components.len(), degree_profiles.len()).unwrap();
    }
    // Rust tuple debug formatting is not TOML array syntax.
    report = report.replace('(', "[").replace(')', "]");
    if let Ok(path) = std::env::var("MOTIF_EXACT_OUTPUT") {
        std::fs::write(path, &report).unwrap();
    }
    for section in report.split("[[dimension_summary]]").skip(1) {
        println!("{}", section.split("[[component]]").next().unwrap());
    }
}

#[test]
fn two_level_population_spacing_variance_has_exact_integer_identity() {
    // N values give N-1 spacings: one normalized spacing N-1, the rest zero.
    for population_size in 2u64..=256 {
        let spacing_count = population_size - 1;
        let squared_deviation_sum = (spacing_count - 1).pow(2) + spacing_count - 1;
        assert_eq!(squared_deviation_sum, spacing_count * (population_size - 2));
        // For N>2, sample variance uses N-2 and instead equals N-1.
        if population_size > 2 {
            assert_eq!(
                squared_deviation_sum / (population_size - 2),
                population_size - 1
            );
        }
    }
}

#[test]
fn modular_rank_detects_singular_and_nonsingular_integer_controls() {
    assert_eq!(modular_rank(vec![vec![1, 1], vec![1, 1]]), 1);
    assert_eq!(modular_rank(vec![vec![0, 1], vec![1, 0]]), 2);
}
