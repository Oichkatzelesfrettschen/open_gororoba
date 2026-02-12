//! Stasheff/associahedron-inspired pentagon residuals.
//!
//! For a 5-tuple `(a,b,c,d,e)`, we compare two extreme parenthesizations:
//! `((((ab)c)d)e)` and `(a(b(c(de))))`.
//! The mismatch serves as a proxy for pentagon identity violation.

use crate::training_data::SEDENION_DIM;

/// Basis-sign element in the sedenion basis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SignedBasis {
    pub basis: usize,
    pub sign: i32,
}

impl SignedBasis {
    fn one() -> Self {
        Self { basis: 0, sign: 1 }
    }
}

/// Result for one 5-ary pentagon comparison.
#[derive(Debug, Clone, Copy)]
pub struct PentagonResidual {
    pub left: SignedBasis,
    pub right: SignedBasis,
    pub residual: f64,
}

/// Evaluate pentagon residual on a given word.
pub fn pentagon_residual(
    word: [usize; 5],
    table: &[[(usize, i32); SEDENION_DIM]; SEDENION_DIM],
) -> PentagonResidual {
    let a = SignedBasis {
        basis: word[0],
        sign: 1,
    };
    let b = SignedBasis {
        basis: word[1],
        sign: 1,
    };
    let c = SignedBasis {
        basis: word[2],
        sign: 1,
    };
    let d = SignedBasis {
        basis: word[3],
        sign: 1,
    };
    let e = SignedBasis {
        basis: word[4],
        sign: 1,
    };

    let left = mul(mul(mul(mul(a, b, table), c, table), d, table), e, table);
    let right = mul(a, mul(b, mul(c, mul(d, e, table), table), table), table);

    let residual = if left == right { 0.0 } else { 1.0 };
    PentagonResidual {
        left,
        right,
        residual,
    }
}

/// Mean pentagon residual over a batch of words.
pub fn mean_pentagon_residual(
    words: &[[usize; 5]],
    table: &[[(usize, i32); SEDENION_DIM]; SEDENION_DIM],
) -> f64 {
    if words.is_empty() {
        return 0.0;
    }
    let sum = words
        .iter()
        .map(|&w| pentagon_residual(w, table).residual)
        .sum::<f64>();
    sum / (words.len() as f64)
}

fn mul(
    lhs: SignedBasis,
    rhs: SignedBasis,
    table: &[[(usize, i32); SEDENION_DIM]; SEDENION_DIM],
) -> SignedBasis {
    if lhs == SignedBasis::one() {
        return rhs;
    }
    if rhs == SignedBasis::one() {
        return lhs;
    }
    let (basis, sign) = table[lhs.basis][rhs.basis];
    SignedBasis {
        basis,
        sign: lhs.sign * rhs.sign * sign,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training_data::build_sedenion_table;

    #[test]
    fn test_pentagon_residual_range() {
        let table = build_sedenion_table();
        let res = pentagon_residual([1, 2, 4, 8, 3], &table);
        assert!(res.residual == 0.0 || res.residual == 1.0);
    }

    #[test]
    fn test_mean_pentagon_residual_bounds() {
        let table = build_sedenion_table();
        let words = [[1, 2, 4, 8, 3], [2, 3, 5, 9, 7], [1, 1, 1, 1, 1]];
        let mean = mean_pentagon_residual(&words, &table);
        assert!((0.0..=1.0).contains(&mean));
    }
}
