/// Compute the F4 Casimir ratio epsilon = C2(26) / |Delta+(F4)|.
///
/// The normalization matches the legacy algebra_experimental API: the 26D
/// fundamental representation has C2(26) = 6 and F4 has 24 positive roots.
pub fn compute_f4_casimir_ratio() -> (f64, usize, f64) {
    let mut positive_roots: Vec<[f64; 4]> = Vec::new();

    for first_basis_index in 0..4 {
        for second_basis_index in (first_basis_index + 1)..4 {
            let mut plus_root = [0.0; 4];
            plus_root[first_basis_index] = 1.0;
            plus_root[second_basis_index] = 1.0;
            positive_roots.push(plus_root);

            let mut minus_root = [0.0; 4];
            minus_root[first_basis_index] = 1.0;
            minus_root[second_basis_index] = -1.0;
            positive_roots.push(minus_root);
        }
    }

    for basis_index in 0..4 {
        let mut short_root = [0.0; 4];
        short_root[basis_index] = 1.0;
        positive_roots.push(short_root);
    }

    for second_sign in [-1.0, 1.0] {
        for third_sign in [-1.0, 1.0] {
            for fourth_sign in [-1.0, 1.0] {
                positive_roots.push([0.5, 0.5 * second_sign, 0.5 * third_sign, 0.5 * fourth_sign]);
            }
        }
    }

    let positive_root_count = positive_roots.len();
    let mut weyl_vector = [0.0; 4];
    for root in &positive_roots {
        for basis_index in 0..4 {
            weyl_vector[basis_index] += 0.5 * root[basis_index];
        }
    }

    let highest_weight = [1.0, 0.0, 0.0, 0.0];
    let mut standard_casimir = 0.0;
    for basis_index in 0..4 {
        standard_casimir += (highest_weight[basis_index] + 2.0 * weyl_vector[basis_index])
            * highest_weight[basis_index];
    }

    let casimir = standard_casimir * 0.5;
    let epsilon = casimir / (positive_root_count as f64);

    (casimir, positive_root_count, epsilon)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computes_f4_casimir_ratio() {
        let (casimir, positive_roots, epsilon) = compute_f4_casimir_ratio();

        assert_eq!(positive_roots, 24);
        assert!((casimir - 6.0).abs() < 1e-9);
        assert!((epsilon - 0.25).abs() < 1e-9);
    }
}
