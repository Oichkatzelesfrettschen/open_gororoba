//! Matrix Representations of Cayley-Dickson Algebras
//!
//! This module provides functions to construct matrix representations of the
//! Quaternions, Octonions, and Sedenions. This is a crucial step for

use crate::cayley_dickson_structs::Sedenion;
use faer::Mat;

// ... (get_quaternion_matrices implementation)

pub fn get_sedenion_matrices() -> Vec<Mat<f64>> {
    let mut basis = [Sedenion::default(); 16];
    for i in 0..16 {
        let mut components = [0.0; 16];
        components[i] = 1.0;
        basis[i] = Sedenion::from_slice(&components);
    }

    let mut matrices = Vec::new();
    for i in 0..16 {
        let mut matrix = Mat::zeros(16, 16);
        for j in 0..16 {
            let res = basis[i] * basis[j];
            let components = res.to_slice();
            for (k, component) in components.iter().enumerate() {
                matrix[(k, j)] = *component;
            }
        }
        matrices.push(matrix);
    }

    matrices
}

#[cfg(test)]
mod tests {
    use super::*;

    // ... (previous tests)

    #[test]
    fn test_sedenion_matrix_representation() {
        let s_mats = get_sedenion_matrices();
        let e1 = &s_mats[1];
        let e9 = &s_mats[9];

        // e1 * e1 = -1
        let e1_sq = e1 * e1;
        assert!((&e1_sq + &s_mats[0]).norm_l2() < 1e-9);

        // e1 * e9 != e9 * e1 (non-commutative)
        let e1e9 = e1 * e9;
        let e9e1 = e9 * e1;
        assert!((&e1e9 - &e9e1).norm_l2() > 1e-9);
    }
}
