use rayon::prelude::*;

use super::{
    arith::{cd_multiply, cd_norm_sq},
    signs::SignTable,
};

/// A zero-divisor candidate found via general-form search.
#[derive(Debug, Clone)]
pub struct GeneralFormZD {
    pub a: Vec<f64>,
    pub b: Vec<f64>,
    pub norm: f64,
    pub blade_order: usize,
}

pub fn find_zero_divisors(dim: usize, atol: f64) -> Vec<(usize, usize, usize, usize, f64)> {
    let mut results = Vec::new();
    for i in 0..dim {
        for j in (i + 1)..dim {
            let mut a = vec![0.0; dim];
            a[i] = 1.0;
            a[j] = 1.0;

            for k in 0..dim {
                for l in (k + 1)..dim {
                    let mut b = vec![0.0; dim];
                    b[k] = 1.0;
                    b[l] = 1.0;
                    let ab = cd_multiply(&a, &b);
                    let norm = cd_norm_sq(&ab).sqrt();
                    if norm < atol {
                        results.push((i, j, k, l, norm));
                    }

                    b[l] = -1.0;
                    let ab = cd_multiply(&a, &b);
                    let norm = cd_norm_sq(&ab).sqrt();
                    if norm < atol {
                        results.push((i, j, k, l, norm));
                    }
                }
            }
        }
    }
    results
}

pub fn find_zero_divisors_3blade(
    dim: usize,
    atol: f64,
) -> Vec<(usize, usize, usize, usize, usize, usize, f64)> {
    let mut results = Vec::new();
    for i in 0..dim {
        for j in (i + 1)..dim {
            for k in (j + 1)..dim {
                let mut a = vec![0.0; dim];
                a[i] = 1.0;
                a[j] = 1.0;
                a[k] = 1.0;

                for l in 0..dim {
                    for m in (l + 1)..dim {
                        for n in (m + 1)..dim {
                            let mut b = vec![0.0; dim];
                            b[l] = 1.0;
                            b[m] = 1.0;
                            b[n] = 1.0;
                            let ab = cd_multiply(&a, &b);
                            let norm = cd_norm_sq(&ab).sqrt();
                            if norm < atol {
                                results.push((i, j, k, l, m, n, norm));
                            }
                        }
                    }
                }
            }
        }
    }
    results
}

pub fn find_zero_divisors_general_form(
    dim: usize,
    n_samples: usize,
    atol: f64,
    seed: u64,
) -> Vec<GeneralFormZD> {
    use rand::{prelude::*, rngs::StdRng};

    let mut rng = StdRng::seed_from_u64(seed);
    let mut results = Vec::new();

    for _ in 0..n_samples {
        let n_components = rng.gen_range(1..=4);

        let mut a = vec![0.0; dim];
        let mut b = vec![0.0; dim];

        let mut a_indices: Vec<usize> = (0..dim).collect();
        a_indices.shuffle(&mut rng);
        for &idx in a_indices.iter().take(n_components) {
            a[idx] = if rng.gen_bool(0.5) { 1.0 } else { -1.0 };
        }

        let mut b_indices: Vec<usize> = (0..dim).collect();
        b_indices.shuffle(&mut rng);
        let b_components = rng.gen_range(1..=4);
        for &idx in b_indices.iter().take(b_components) {
            b[idx] = if rng.gen_bool(0.5) { 1.0 } else { -1.0 };
        }

        let ab = cd_multiply(&a, &b);
        let norm = cd_norm_sq(&ab).sqrt();

        if norm < atol {
            results.push(GeneralFormZD {
                a: a.clone(),
                b: b.clone(),
                norm,
                blade_order: n_components,
            });
        }
    }

    results
}

pub fn count_pathion_zero_divisors(
    n_general_samples: usize,
    atol: f64,
    seed: u64,
) -> (usize, usize, usize) {
    let dim = 32;
    let zd_2blade = find_zero_divisors_parallel(dim, atol);
    let zd_general = find_zero_divisors_general_form(dim, n_general_samples, atol, seed);
    let zd_3blade_count = zd_general.iter().filter(|z| z.blade_order == 3).count();

    (zd_2blade.len(), zd_3blade_count, zd_general.len())
}

pub fn zd_spectrum_analysis(
    dim: usize,
    n_samples: usize,
    n_bins: usize,
    seed: u64,
) -> (f64, f64, f64, Vec<usize>) {
    use rand::{prelude::*, rngs::StdRng};

    let mut rng = StdRng::seed_from_u64(seed);
    let mut norms = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        let a: Vec<f64> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let b: Vec<f64> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let ab = cd_multiply(&a, &b);
        norms.push(cd_norm_sq(&ab).sqrt());
    }

    let min_norm = norms.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_norm = norms.iter().cloned().fold(0.0, f64::max);
    let mean_norm = norms.iter().sum::<f64>() / n_samples as f64;

    let mut histogram = vec![0usize; n_bins];
    let bin_width = (max_norm - min_norm) / n_bins as f64;
    if bin_width > 0.0 {
        for &norm in &norms {
            let bin = ((norm - min_norm) / bin_width).floor() as usize;
            histogram[bin.min(n_bins - 1)] += 1;
        }
    } else {
        histogram[0] = n_samples;
    }

    (min_norm, max_norm, mean_norm, histogram)
}

pub fn find_zero_divisors_parallel(
    dim: usize,
    atol: f64,
) -> Vec<(usize, usize, usize, usize, f64)> {
    let sign_table = SignTable::new(dim);
    let pairs: Vec<(usize, usize)> = (0..dim)
        .flat_map(|i| ((i + 1)..dim).map(move |j| (i, j)))
        .collect();

    pairs
        .par_iter()
        .flat_map(|&(i, j)| {
            let mut results = Vec::new();
            for k in 0..dim {
                for l in (k + 1)..dim {
                    if two_blade_product_is_zero(&sign_table, i, j, k, l, 1) && 0.0 < atol {
                        results.push((i, j, k, l, 0.0));
                    }
                    if two_blade_product_is_zero(&sign_table, i, j, k, l, -1) && 0.0 < atol {
                        results.push((i, j, k, l, 0.0));
                    }
                }
            }
            results
        })
        .collect()
}

#[inline(always)]
fn two_blade_product_is_zero(
    sign_table: &SignTable,
    i: usize,
    j: usize,
    k: usize,
    l: usize,
    right_sign: i32,
) -> bool {
    let mut basis_terms = [usize::MAX; 4];
    let mut coeffs = [0i32; 4];
    let mut used = 0usize;

    accumulate_sparse_basis_term(
        &mut basis_terms,
        &mut coeffs,
        &mut used,
        i ^ k,
        sign_table.sign(i, k),
    );
    accumulate_sparse_basis_term(
        &mut basis_terms,
        &mut coeffs,
        &mut used,
        i ^ l,
        right_sign * sign_table.sign(i, l),
    );
    accumulate_sparse_basis_term(
        &mut basis_terms,
        &mut coeffs,
        &mut used,
        j ^ k,
        sign_table.sign(j, k),
    );
    accumulate_sparse_basis_term(
        &mut basis_terms,
        &mut coeffs,
        &mut used,
        j ^ l,
        right_sign * sign_table.sign(j, l),
    );

    coeffs[..used].iter().all(|&coeff| coeff == 0)
}

#[inline(always)]
fn accumulate_sparse_basis_term(
    basis_terms: &mut [usize; 4],
    coeffs: &mut [i32; 4],
    used: &mut usize,
    basis: usize,
    coeff: i32,
) {
    for slot in 0..*used {
        if basis_terms[slot] == basis {
            coeffs[slot] += coeff;
            return;
        }
    }

    basis_terms[*used] = basis;
    coeffs[*used] = coeff;
    *used += 1;
}

/// Koebisu's D_2 polynomial for zero-divisor detection.
///
/// For a sedenion v = v_1 + v_2*e_8 (where v_1 = v[0..8], v_2 = v[8..16]):
///
///   D_2(v) = (||v_1||^2 - ||v_2||^2)^2 + 4*<v_1, v_2>^2
///
/// A nonzero v is a zero-divisor iff D_2(v) = 0.
///
/// This replaces the O(N^3) matrix determinant with an O(N) polynomial.
///
/// Reference: Koebisu (arXiv:2512.13002), Theorem 3.6, Lemma 3.3.
#[inline]
pub fn koebisu_d2(v: &[f64]) -> f64 {
    debug_assert!(v.len() >= 16, "Koebisu D_2 requires at least 16 components");
    let half = v.len() / 2;

    let mut norm_sq_v1 = 0.0_f64;
    let mut norm_sq_v2 = 0.0_f64;
    let mut dot_v1_v2 = 0.0_f64;

    for i in 0..half {
        let x = v[i];
        let y = v[i + half];
        norm_sq_v1 += x * x;
        norm_sq_v2 += y * y;
        dot_v1_v2 += x * y;
    }

    let b = norm_sq_v1 - norm_sq_v2;
    b * b + 4.0 * dot_v1_v2 * dot_v1_v2
}

/// Fast zero-divisor membership test using Koebisu's D_2 polynomial.
///
/// Returns true if v is a zero-divisor (D_2(v) < epsilon).
/// O(N) time, zero allocation, branchless inner loop.
#[inline]
pub fn is_zero_divisor_koebisu(v: &[f64], epsilon: f64) -> bool {
    koebisu_d2(v) < epsilon
}

/// D_1 polynomial: the squared norm.
///
///   D_1(v) = ||v_1||^2 + ||v_2||^2 = ||v||^2
///
/// The full determinant factorization is det(L_v) = D_1(v)^4 * D_2(v)^2.
#[inline]
pub fn koebisu_d1(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum()
}

/// Gourlay/Gresnigt epsilon automorphism (order 2).
///
///   epsilon(A + B*e_8) = A - B*e_8
///
/// Flips the sign of the upper octonion half.
/// Reference: Gourlay & Gresnigt (arXiv:2407.01580), Eq 4.
#[inline]
pub fn gourlay_epsilon(v: &[f64; 16]) -> [f64; 16] {
    let mut out = *v;
    for x in out[8..].iter_mut() {
        *x = -*x;
    }
    out
}

/// Gourlay/Gresnigt psi automorphism (order 3).
///
///   psi(A + B*e_8) = (1/4)[A + 3A* + sqrt(3)(B - B*)]
///                   + (1/4)[B + 3B* - sqrt(3)(A - A*)] * e_8
///
/// where A* is the octonion conjugate of A (negate imaginary, keep real).
///
/// This cycles the three generations: psi^3 = Id.
/// Reference: Gourlay & Gresnigt (arXiv:2407.01580), Eq 5.
pub fn gourlay_psi(v: &[f64; 16]) -> [f64; 16] {
    let sqrt3 = 3.0_f64.sqrt();

    // Split into octonion halves
    let a = &v[..8];   // lower octonion
    let b = &v[8..];   // upper octonion

    // Octonion conjugation: negate imaginary (indices 1..7), keep real (index 0)
    let mut a_conj = [0.0_f64; 8];
    a_conj[0] = a[0];
    for i in 1..8 { a_conj[i] = -a[i]; }

    let mut b_conj = [0.0_f64; 8];
    b_conj[0] = b[0];
    for i in 1..8 { b_conj[i] = -b[i]; }

    let mut out = [0.0_f64; 16];

    // Lower half: (1/4)[A + 3A* + sqrt(3)(B - B*)]
    for i in 0..8 {
        out[i] = 0.25 * (a[i] + 3.0 * a_conj[i] + sqrt3 * (b[i] - b_conj[i]));
    }

    // Upper half: (1/4)[B + 3B* - sqrt(3)(A - A*)]
    for i in 0..8 {
        out[8 + i] = 0.25 * (b[i] + 3.0 * b_conj[i] - sqrt3 * (a[i] - a_conj[i]));
    }

    out
}

/// Apply psi n times (for computing psi^2, psi^3, etc.).
pub fn gourlay_psi_n(v: &[f64; 16], n: usize) -> [f64; 16] {
    let mut result = *v;
    for _ in 0..n {
        result = gourlay_psi(&result);
    }
    result
}

/// Cross-generational signed friction between subalgebra O_i and O_j.
///
/// Measures how much topological friction exists BETWEEN generations
/// by computing the associator [A_rot, X, Y] where X is in O_i and Y
/// is in O_j.
pub fn cross_generational_friction(
    mode_a: usize, mode_b: usize,
    subalgebra_i: &[usize], subalgebra_j: &[usize],
) -> f64 {
    use super::arith::cd_multiply;

    let dim = 16_usize;
    let theta = std::f64::consts::FRAC_PI_4;

    // Construct the rotated braid element
    let mut a_rot = vec![0.0; dim];
    a_rot[mode_a] = theta.cos();
    a_rot[mode_b] = theta.sin();

    let mut total = 0.0_f64;

    for &x_idx in subalgebra_i {
        if x_idx == 0 || x_idx == mode_a || x_idx == mode_b { continue; }
        for &y_idx in subalgebra_j {
            if y_idx == 0 || y_idx == mode_a || y_idx == mode_b { continue; }
            if x_idx == y_idx { continue; }

            let mut ex = vec![0.0; dim]; ex[x_idx] = 1.0;
            let mut ey = vec![0.0; dim]; ey[y_idx] = 1.0;

            // [A_rot, X, Y] = (A_rot * X) * Y - A_rot * (X * Y)
            let ax = cd_multiply(&a_rot, &ex);
            let axy = cd_multiply(&ax, &ey);
            let xy = cd_multiply(&ex, &ey);
            let a_xy = cd_multiply(&a_rot, &xy);

            // Signed sum of the associator
            for k in 0..dim {
                total += axy[k] - a_xy[k];
            }
        }
    }
    total
}
