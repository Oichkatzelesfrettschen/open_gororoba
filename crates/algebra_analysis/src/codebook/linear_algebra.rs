//! Pure linear-algebra helpers used by codebook coupling analysis.
//!
//! All functions operate on `&[f64]` / `[f64; 8]` / `Vec<Vec<f64>>`
//! and carry no dependency on the codebook's type universe
//! (TypedCarrier, CarrierSet, EncodingDictionary, etc.).
//!
//! Numerical-stability strategy:
//!   * `kahan_dot` / `kahan_norm_sq` use Kahan compensated summation
//!     so high-rank Gram-Schmidt passes do not accumulate rounding
//!     error.
//!   * `gram_schmidt_basis` produces an orthonormal column-space
//!     basis for up to 8 input vectors; pivot tolerance 1e-10 on the
//!     normalized result.
//!   * `find_pivot_columns_reduced` runs a separate Gram-Schmidt on
//!     the already-projected reduced-space columns to identify the
//!     column-pivot set; tolerance 1e-8 on the *unnormalized* norm
//!     squared because the projections may already be small.
//!   * `invert_nxn` and `det_nxn` use partial-pivot Gauss-Jordan /
//!     LU with absolute tolerance 1e-12 on the pivot magnitude
//!     before declaring the system singular.
//!
//! All items `pub(super)`.

pub(super) fn kahan_dot(a: &[f64], b: &[f64]) -> f64 {
    let mut sum = 0.0_f64;
    let mut c = 0.0_f64;
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        let y = ai * bi - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    sum
}

/// Kahan compensated squared norm.
pub(super) fn kahan_norm_sq(v: &[f64]) -> f64 {
    let mut sum = 0.0_f64;
    let mut c = 0.0_f64;
    for &x in v {
        let y = x * x - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    sum
}

/// Build an orthonormal basis for the column space of the given vectors.
/// Returns (basis_vectors, rank).
pub(super) fn gram_schmidt_basis(vectors: &[[f64; 8]]) -> (Vec<[f64; 8]>, usize) {
    let mut basis: Vec<[f64; 8]> = Vec::with_capacity(8);

    for v in vectors {
        if basis.len() == 8 {
            break;
        }
        let mut w = *v;
        for b in &basis {
            let dot = kahan_dot(&w, b);
            for (wk, &bk) in w.iter_mut().zip(b.iter()) {
                *wk -= dot * bk;
            }
        }
        let norm = kahan_norm_sq(&w).sqrt();
        if norm > 1e-10 {
            w.iter_mut().for_each(|x| *x /= norm);
            basis.push(w);
        }
    }

    let rank = basis.len();
    (basis, rank)
}

/// Project an 8D vector onto the orthonormal basis, giving an r-dimensional vector.
pub(super) fn project_to_basis(v: &[f64; 8], basis: &[[f64; 8]]) -> Vec<f64> {
    basis.iter().map(|b| kahan_dot(v, b)).collect()
}

/// Find r linearly independent columns from reduced-space vectors.
pub(super) fn find_pivot_columns_reduced(phi_r: &[Vec<f64>], rank: usize) -> Vec<usize> {
    let mut pivots = Vec::with_capacity(rank);
    let mut basis = Vec::<Vec<f64>>::with_capacity(rank);

    for (c, col) in phi_r.iter().enumerate() {
        if pivots.len() == rank {
            break;
        }

        let mut v = col.clone();
        for b in &basis {
            let dot = kahan_dot(&v, b);
            let norm_sq = kahan_norm_sq(b);
            if norm_sq > 1e-12 {
                for (vk, &bk) in v.iter_mut().zip(b.iter()) {
                    *vk -= (dot / norm_sq) * bk;
                }
            }
        }

        let norm_sq = kahan_norm_sq(&v);
        if norm_sq > 1e-8 {
            basis.push(v);
            pivots.push(c);
        }
    }

    pivots
}

/// Build an r x r matrix from selected columns.
pub(super) fn build_square_matrix(
    phi_r: &[Vec<f64>],
    pivot_cols: &[usize],
    rank: usize,
) -> Vec<Vec<f64>> {
    let mut m = vec![vec![0.0; rank]; rank];
    for (j, &c) in pivot_cols.iter().enumerate() {
        for (i, row) in m.iter_mut().enumerate() {
            row[j] = phi_r[c][i];
        }
    }
    m
}

/// Invert an n x n matrix using Gauss-Jordan elimination.
pub(super) fn invert_nxn(m: &[Vec<f64>], n: usize) -> Option<Vec<Vec<f64>>> {
    let nn = 2 * n;
    let mut aug: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row = vec![0.0; nn];
            row[..n].copy_from_slice(&m[i]);
            row[n + i] = 1.0;
            row
        })
        .collect();

    for col in 0..n {
        let max_row = aug[col..]
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a[col].abs().partial_cmp(&b[col].abs()).unwrap())
            .map(|(idx, _)| idx + col)
            .unwrap();

        if aug[max_row][col].abs() < 1e-12 {
            return None;
        }

        aug.swap(col, max_row);

        let pivot = aug[col][col];
        aug[col].iter_mut().for_each(|v| *v /= pivot);

        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[row][col];
            let pivot_row: Vec<f64> = aug[col].clone();
            for (v, &p) in aug[row].iter_mut().zip(pivot_row.iter()) {
                *v -= factor * p;
            }
        }
    }

    Some(aug.iter().map(|row| row[n..].to_vec()).collect())
}

/// Multiply two n x n matrices.
pub(super) fn mat_mul_nxn(a: &[Vec<f64>], b: &[Vec<f64>], n: usize) -> Vec<Vec<f64>> {
    let mut c = vec![vec![0.0; n]; n];
    for (c_row, a_row) in c.iter_mut().zip(a.iter()) {
        for (j, c_val) in c_row.iter_mut().enumerate() {
            *c_val = a_row
                .iter()
                .zip(b.iter())
                .map(|(&a_ik, b_row)| a_ik * b_row[j])
                .sum();
        }
    }
    c
}

/// Compute the determinant of an n x n matrix via LU decomposition.
pub(super) fn det_nxn(m: &[Vec<f64>], n: usize) -> f64 {
    let mut a: Vec<Vec<f64>> = m.to_vec();
    let mut sign = 1.0f64;

    for col in 0..n {
        let max_row = a[col..]
            .iter()
            .enumerate()
            .max_by(|(_, ra), (_, rb)| ra[col].abs().partial_cmp(&rb[col].abs()).unwrap())
            .map(|(idx, _)| idx + col)
            .unwrap();

        if a[max_row][col].abs() < 1e-12 {
            return 0.0;
        }

        if max_row != col {
            a.swap(col, max_row);
            sign = -sign;
        }

        let pivot = a[col][col];
        for row in (col + 1)..n {
            let factor = a[row][col] / pivot;
            let pivot_row: Vec<f64> = a[col].clone();
            for (v, &p) in a[row].iter_mut().zip(pivot_row.iter()).skip(col) {
                *v -= factor * p;
            }
        }
    }

    sign * a.iter().enumerate().map(|(i, row)| row[i]).product::<f64>()
}
