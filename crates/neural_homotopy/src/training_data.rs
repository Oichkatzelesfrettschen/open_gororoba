//! Training data generation for sedenion multiplication synthesis.

use algebra_core::construction::cayley_dickson::cd_basis_mul_sign;

pub const SEDENION_DIM: usize = 16;

/// One supervised sample extracted from basis multiplication.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MultiplicationSample {
    pub lhs: usize,
    pub rhs: usize,
    pub product_basis: usize,
    pub product_sign: i32,
}

/// Build dense basis multiplication table for sedenions.
///
/// Entry `[i][j]` is `(basis, sign)` for `e_i * e_j = sign * e_basis`.
pub fn build_sedenion_table() -> [[(usize, i32); SEDENION_DIM]; SEDENION_DIM] {
    let mut table = [[(0usize, 1i32); SEDENION_DIM]; SEDENION_DIM];
    for (i, row) in table.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            let sign = cd_basis_mul_sign(SEDENION_DIM, i, j);
            let basis = i ^ j;
            *cell = (basis, sign);
        }
    }
    table
}

/// Flatten multiplication table into supervised samples.
pub fn multiplication_samples() -> Vec<MultiplicationSample> {
    let table = build_sedenion_table();
    let mut out = Vec::with_capacity(SEDENION_DIM * SEDENION_DIM);
    for (i, row) in table.iter().enumerate() {
        for (j, &(basis, sign)) in row.iter().enumerate() {
            out.push(MultiplicationSample {
                lhs: i,
                rhs: j,
                product_basis: basis,
                product_sign: sign,
            });
        }
    }
    out
}

/// Encode `(lhs, rhs)` into a token id in `[0, 255]`.
pub fn encode_pair(lhs: usize, rhs: usize) -> usize {
    lhs * SEDENION_DIM + rhs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_table_dimensions() {
        let table = build_sedenion_table();
        assert_eq!(table.len(), SEDENION_DIM);
        assert_eq!(table[0].len(), SEDENION_DIM);
    }

    #[test]
    fn test_samples_cover_full_basis_product_space() {
        let samples = multiplication_samples();
        assert_eq!(samples.len(), SEDENION_DIM * SEDENION_DIM);
    }

    #[test]
    fn test_encode_pair_bounds() {
        assert_eq!(encode_pair(0, 0), 0);
        assert_eq!(encode_pair(15, 15), 255);
    }
}
