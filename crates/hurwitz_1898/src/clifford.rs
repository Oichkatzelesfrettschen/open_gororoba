//! Clifford algebra structure (Hurwitz Part I.5, eq.5-9).
//!
//! The composition identity at dimension n requires n-1 real n x n
//! skew-symmetric matrices B_1, ..., B_{n-1} satisfying:
//!   B_i^2 = -I            (squares to minus identity)
//!   B_i * B_k = -B_k * B_i  (anticommutativity, i != k)
//!   B_i' = -B_i           (skew-symmetric)
//!
//! These are the Clifford algebra relations for Cl(n-1, 0) over R.
//!
//! Mirrors: HurwitzTheorem.v Part I.5, Module Type CliffordSystem.

/// Check whether n-1 real n x n matrices satisfy the Clifford relations.
///
/// Each matrix is given as a flat n x n array in row-major order.
/// Returns (squares_ok, anticomm_ok, skew_ok).
///
/// Mirrors: HurwitzTheorem.v eq.9 (B_i^2 = -I, B_i*B_k = -B_k*B_i).
pub fn check_clifford_relations(n: usize, matrices: &[Vec<f64>], atol: f64) -> (bool, bool, bool) {
    let num = matrices.len();
    assert!(matrices.iter().all(|m| m.len() == n * n));

    // Check B_i^2 = -I
    let squares_ok = matrices.iter().all(|b| {
        let b2 = mat_mul(n, b, b);
        let mut neg_i = vec![0.0; n * n];
        for k in 0..n {
            neg_i[k * n + k] = -1.0;
        }
        mat_close(&b2, &neg_i, atol)
    });

    // Check B_i * B_k = -B_k * B_i for i != k
    let mut anticomm_ok = true;
    for i in 0..num {
        for k in (i + 1)..num {
            let bik = mat_mul(n, &matrices[i], &matrices[k]);
            let bki = mat_mul(n, &matrices[k], &matrices[i]);
            let neg_bki: Vec<f64> = bki.iter().map(|x| -x).collect();
            if !mat_close(&bik, &neg_bki, atol) {
                anticomm_ok = false;
            }
        }
    }

    // Check B_i' = -B_i (skew-symmetric)
    let skew_ok = matrices.iter().all(|b| {
        for r in 0..n {
            for c in 0..n {
                if (b[r * n + c] + b[c * n + r]).abs() > atol {
                    return false;
                }
            }
        }
        true
    });

    (squares_ok, anticomm_ok, skew_ok)
}

/// Extract the B_i matrices from the quaternion multiplication table.
/// Returns 3 matrices B_1, B_2, B_3 (each 4x4).
///
/// B_i = A_i * A_4' where A_i is the i-th column of the multiplication table.
/// For quaternions: these are the matrices for left-multiplication by i, j, k.
///
/// Mirrors: HurwitzTheorem.v eq.7, hurwitz_A0_dim4_valid.
pub fn quaternion_clifford_generators() -> Vec<Vec<f64>> {
    // Left-multiplication by i: q -> i*q
    // i*1 = i, i*i = -1, i*j = k, i*k = -j
    #[rustfmt::skip]
    let b1 = vec![
        0.0, -1.0,  0.0,  0.0,
        1.0,  0.0,  0.0,  0.0,
        0.0,  0.0,  0.0, -1.0,
        0.0,  0.0,  1.0,  0.0,
    ];
    // Left-multiplication by j: q -> j*q
    #[rustfmt::skip]
    let b2 = vec![
        0.0,  0.0, -1.0,  0.0,
        0.0,  0.0,  0.0,  1.0,
        1.0,  0.0,  0.0,  0.0,
        0.0, -1.0,  0.0,  0.0,
    ];
    // Left-multiplication by k: q -> k*q
    #[rustfmt::skip]
    let b3 = vec![
        0.0,  0.0,  0.0, -1.0,
        0.0,  0.0, -1.0,  0.0,
        0.0,  1.0,  0.0,  0.0,
        1.0,  0.0,  0.0,  0.0,
    ];
    vec![b1, b2, b3]
}

// --- Internal matrix helpers ---

fn mat_mul(n: usize, a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut c = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                c[i * n + j] += a[i * n + k] * b[k * n + j];
            }
        }
    }
    c
}

fn mat_close(a: &[f64], b: &[f64], atol: f64) -> bool {
    a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < atol)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quaternion_clifford_relations() {
        let gens = quaternion_clifford_generators();
        let (sq, ac, sk) = check_clifford_relations(4, &gens, 1e-10);
        assert!(sq, "B_i^2 = -I should hold for quaternion generators");
        assert!(ac, "B_i*B_k = -B_k*B_i should hold");
        assert!(sk, "B_i should be skew-symmetric");
    }

    #[test]
    fn test_product_count() {
        // 3 generators -> 2^3 = 8 products (including identity)
        // These are: I, B1, B2, B3, B1B2, B1B3, B2B3, B1B2B3
        let gens = quaternion_clifford_generators();
        let n = 4_usize;

        // Verify B1*B2*B3 squares to +I or -I
        let b12 = mat_mul(n, &gens[0], &gens[1]);
        let b123 = mat_mul(n, &b12, &gens[2]);
        let b123_sq = mat_mul(n, &b123, &b123);

        // (B1*B2*B3)^2 = B1*B2*B3*B1*B2*B3
        // For n=4 (3 generators): the full product squares to +I or -I.
        // Check it's one of them.
        let mut pos_i = vec![0.0; 16];
        let mut neg_i = vec![0.0; 16];
        for k in 0..4 {
            pos_i[k * 4 + k] = 1.0;
            neg_i[k * 4 + k] = -1.0;
        }
        assert!(
            mat_close(&b123_sq, &pos_i, 1e-10) || mat_close(&b123_sq, &neg_i, 1e-10),
            "(B1*B2*B3)^2 should be +I or -I"
        );
    }
}
