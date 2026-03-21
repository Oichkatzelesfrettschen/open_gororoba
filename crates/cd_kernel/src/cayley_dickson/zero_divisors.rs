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
