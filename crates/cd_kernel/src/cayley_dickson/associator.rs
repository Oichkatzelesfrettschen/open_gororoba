use rayon::prelude::*;

use super::arith::{allclose, cd_multiply, cd_norm_sq};

#[inline]
pub fn cd_associator(a: &[f64], b: &[f64], c: &[f64]) -> Vec<f64> {
    let ab = cd_multiply(a, b);
    let abc_left = cd_multiply(&ab, c);
    let bc = cd_multiply(b, c);
    let abc_right = cd_multiply(a, &bc);

    abc_left
        .iter()
        .zip(&abc_right)
        .map(|(l, r)| l - r)
        .collect()
}

#[inline]
pub fn cd_associator_norm(a: &[f64], b: &[f64], c: &[f64]) -> f64 {
    let assoc = cd_associator(a, b, c);
    cd_norm_sq(&assoc).sqrt()
}

#[inline]
fn simd_available_x86_64() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

pub fn batch_associator_norms(
    a_flat: &[f64],
    b_flat: &[f64],
    c_flat: &[f64],
    dim: usize,
    n_triples: usize,
) -> Vec<f64> {
    debug_assert_eq!(a_flat.len(), dim * n_triples);
    debug_assert_eq!(b_flat.len(), dim * n_triples);
    debug_assert_eq!(c_flat.len(), dim * n_triples);

    let mut results = Vec::with_capacity(n_triples);
    for i in 0..n_triples {
        let start = i * dim;
        let end = start + dim;
        results.push(cd_associator_norm(
            &a_flat[start..end],
            &b_flat[start..end],
            &c_flat[start..end],
        ));
    }
    results
}

pub fn batch_associator_norms_sq(
    a_flat: &[f64],
    b_flat: &[f64],
    c_flat: &[f64],
    dim: usize,
    n_triples: usize,
) -> Vec<f64> {
    debug_assert_eq!(a_flat.len(), dim * n_triples);
    debug_assert_eq!(b_flat.len(), dim * n_triples);
    debug_assert_eq!(c_flat.len(), dim * n_triples);

    let mut results = Vec::with_capacity(n_triples);
    for i in 0..n_triples {
        let start = i * dim;
        let end = start + dim;
        let assoc = cd_associator(
            &a_flat[start..end],
            &b_flat[start..end],
            &c_flat[start..end],
        );
        results.push(cd_norm_sq(&assoc));
    }
    results
}

pub fn batch_associator_norms_parallel(
    a_flat: &[f64],
    b_flat: &[f64],
    c_flat: &[f64],
    dim: usize,
    n_triples: usize,
) -> Vec<f64> {
    debug_assert_eq!(a_flat.len(), dim * n_triples);
    debug_assert_eq!(b_flat.len(), dim * n_triples);
    debug_assert_eq!(c_flat.len(), dim * n_triples);

    (0..n_triples)
        .into_par_iter()
        .map(|i| {
            let start = i * dim;
            let end = start + dim;
            cd_associator_norm(
                &a_flat[start..end],
                &b_flat[start..end],
                &c_flat[start..end],
            )
        })
        .collect()
}

/// Batch compute sedenion associator norms for a sequence of 16D vectors.
///
/// Uses sliding window of 3: Assoc(v[i], v[i+1], v[i+2]).
/// Optimized with AVX2+FMA if detected at runtime.
pub fn batch_sedenion_associator_norms(vectors: &[[f64; 16]]) -> Vec<f64> {
    if vectors.len() < 3 {
        return Vec::new();
    }

    let use_simd = simd_available_x86_64();

    (0..vectors.len() - 2)
        .map(|i| {
            if use_simd {
                super::simd::sedenion_associator_norm_sq_flat(
                    &vectors[i],
                    &vectors[i + 1],
                    &vectors[i + 2],
                )
                .sqrt()
            } else {
                cd_associator_norm(&vectors[i], &vectors[i + 1], &vectors[i + 2])
            }
        })
        .collect()
}

/// Parallel batch compute sedenion associator norms for a sequence of 16D vectors.
///
/// Uses Rayon for multi-core scaling and AVX2+FMA for per-core throughput.
pub fn batch_sedenion_associator_norms_parallel(vectors: &[[f64; 16]]) -> Vec<f64> {
    if vectors.len() < 3 {
        return Vec::new();
    }

    let use_simd = simd_available_x86_64();

    (0..vectors.len() - 2)
        .into_par_iter()
        .map(|i| {
            if use_simd {
                super::simd::sedenion_associator_norm_sq_flat(
                    &vectors[i],
                    &vectors[i + 1],
                    &vectors[i + 2],
                )
                .sqrt()
            } else {
                cd_associator_norm(&vectors[i], &vectors[i + 1], &vectors[i + 2])
            }
        })
        .collect()
}

/// Batch compute associator norms for a sequence of vectors of arbitrary
/// power-of-2 dimension using a sliding window of 3.
///
/// For dim=16, delegates to the SIMD-accelerated sedenion path.
/// For all other dims, uses the generic recursive `cd_associator_norm`.
///
/// Returns `n - 2` norms for `n` input vectors (one per consecutive triple).
pub fn batch_sliding_associator_norms(vectors: &[Vec<f64>], dim: usize) -> Vec<f64> {
    assert!(dim.is_power_of_two(), "dim must be a power of 2, got {dim}");
    if vectors.len() < 3 {
        return Vec::new();
    }
    debug_assert!(vectors.iter().all(|v| v.len() == dim));

    let use_simd = simd_available_x86_64();

    if dim == 16 && use_simd {
        let fixed: Vec<[f64; 16]> = vectors
            .iter()
            .map(|v| {
                let mut arr = [0.0; 16];
                arr.copy_from_slice(v);
                arr
            })
            .collect();
        return batch_sedenion_associator_norms(&fixed);
    }

    if dim == 32 && use_simd {
        return (0..vectors.len() - 2)
            .map(|i| {
                let a: [f64; 32] = vectors[i].as_slice().try_into().unwrap();
                let b: [f64; 32] = vectors[i + 1].as_slice().try_into().unwrap();
                let c: [f64; 32] = vectors[i + 2].as_slice().try_into().unwrap();
                super::simd::pathion_associator_norm_sq_flat(&a, &b, &c).sqrt()
            })
            .collect();
    }

    if dim == 64 && use_simd {
        return (0..vectors.len() - 2)
            .map(|i| {
                let a: [f64; 64] = vectors[i].as_slice().try_into().unwrap();
                let b: [f64; 64] = vectors[i + 1].as_slice().try_into().unwrap();
                let c: [f64; 64] = vectors[i + 2].as_slice().try_into().unwrap();
                super::simd::chingon_associator_norm_sq_flat(&a, &b, &c).sqrt()
            })
            .collect();
    }

    (0..vectors.len() - 2)
        .map(|i| cd_associator_norm(&vectors[i], &vectors[i + 1], &vectors[i + 2]))
        .collect()
}

/// Parallel variant of [`batch_sliding_associator_norms`].
///
/// Uses Rayon for multi-core scaling. For dim=16, 32, and 64, leverages
/// AVX2+FMA SIMD via the sedenion/pathion/chingon fast paths.
pub fn batch_sliding_associator_norms_parallel(vectors: &[Vec<f64>], dim: usize) -> Vec<f64> {
    assert!(dim.is_power_of_two(), "dim must be a power of 2, got {dim}");
    if vectors.len() < 3 {
        return Vec::new();
    }
    debug_assert!(vectors.iter().all(|v| v.len() == dim));

    let use_simd = simd_available_x86_64();

    if dim == 16 && use_simd {
        let fixed: Vec<[f64; 16]> = vectors
            .iter()
            .map(|v| {
                let mut arr = [0.0; 16];
                arr.copy_from_slice(v);
                arr
            })
            .collect();
        return batch_sedenion_associator_norms_parallel(&fixed);
    }

    if dim == 32 && use_simd {
        return (0..vectors.len() - 2)
            .into_par_iter()
            .map(|i| {
                let a: [f64; 32] = vectors[i].as_slice().try_into().unwrap();
                let b: [f64; 32] = vectors[i + 1].as_slice().try_into().unwrap();
                let c: [f64; 32] = vectors[i + 2].as_slice().try_into().unwrap();
                super::simd::pathion_associator_norm_sq_flat(&a, &b, &c).sqrt()
            })
            .collect();
    }

    if dim == 64 && use_simd {
        return (0..vectors.len() - 2)
            .into_par_iter()
            .map(|i| {
                let a: [f64; 64] = vectors[i].as_slice().try_into().unwrap();
                let b: [f64; 64] = vectors[i + 1].as_slice().try_into().unwrap();
                let c: [f64; 64] = vectors[i + 2].as_slice().try_into().unwrap();
                super::simd::chingon_associator_norm_sq_flat(&a, &b, &c).sqrt()
            })
            .collect();
    }

    (0..vectors.len() - 2)
        .into_par_iter()
        .map(|i| cd_associator_norm(&vectors[i], &vectors[i + 1], &vectors[i + 2]))
        .collect()
}

/// SIMD-optimized octonion associator using flat AVX2 multiply.
#[inline]
pub fn octonion_associator_simd(a: &[f64; 8], b: &[f64; 8], c: &[f64; 8]) -> [f64; 8] {
    use super::simd::octonion_multiply_flat;
    let ab = octonion_multiply_flat(a, b);
    let ab_c = octonion_multiply_flat(&ab, c);
    let bc = octonion_multiply_flat(b, c);
    let a_bc = octonion_multiply_flat(a, &bc);
    let mut assoc = [0.0_f64; 8];
    for k in 0..8 {
        assoc[k] = ab_c[k] - a_bc[k];
    }
    assoc
}

/// Batch SIMD octonion associator for all 210 ordered basis triples.
pub fn batch_octonion_basis_associators() -> Vec<(usize, usize, usize, [f64; 8])> {
    let mut results = Vec::with_capacity(210);
    for i in 1..8 {
        for j in 1..8 {
            if j == i { continue; }
            for k in 1..8 {
                if k == i || k == j { continue; }
                let mut ei = [0.0_f64; 8]; ei[i] = 1.0;
                let mut ej = [0.0_f64; 8]; ej[j] = 1.0;
                let mut ek = [0.0_f64; 8]; ek[k] = 1.0;
                let assoc = octonion_associator_simd(&ei, &ej, &ek);
                results.push((i, j, k, assoc));
            }
        }
    }
    results
}

pub fn measure_associator_density(dim: usize, trials: usize, seed: u64, atol: f64) -> (f64, usize) {
    use rand::{prelude::*, rngs::StdRng};

    let mut rng = StdRng::seed_from_u64(seed);
    let mut failures = 0;

    for _ in 0..trials {
        let a: Vec<f64> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let b: Vec<f64> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let c: Vec<f64> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();

        let ab = cd_multiply(&a, &b);
        let abc1 = cd_multiply(&ab, &c);
        let bc = cd_multiply(&b, &c);
        let abc2 = cd_multiply(&a, &bc);

        if !allclose(&abc1, &abc2, atol) {
            failures += 1;
        }
    }

    (failures as f64 / trials as f64 * 100.0, failures)
}

/// Result of Monte Carlo associator independence statistics for one dimension.
#[derive(Debug, Clone)]
pub struct AssociatorStats {
    pub dim: usize,
    pub n_trials: usize,
    pub mean_assoc_sq: f64,
    pub std_assoc_sq: f64,
    pub mean_ab_c_sq: f64,
    pub mean_a_bc_sq: f64,
    pub mean_cross_term: f64,
    pub correlation_coeff: f64,
}

pub fn associator_independence_stats(dim: usize, n_trials: usize, seed: u64) -> AssociatorStats {
    use rand::prelude::*;
    use rand_chacha::ChaCha8Rng;

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut assoc_sq = Vec::with_capacity(n_trials);
    let mut ab_c_sq = Vec::with_capacity(n_trials);
    let mut a_bc_sq = Vec::with_capacity(n_trials);
    let mut cross = Vec::with_capacity(n_trials);

    for _ in 0..n_trials {
        let a = random_unit_vec(dim, &mut rng);
        let b = random_unit_vec(dim, &mut rng);
        let c = random_unit_vec(dim, &mut rng);

        let ab = cd_multiply(&a, &b);
        let bc = cd_multiply(&b, &c);
        let ab_c_vec = cd_multiply(&ab, &c);
        let a_bc_vec = cd_multiply(&a, &bc);

        let assoc_vec: Vec<f64> = ab_c_vec.iter().zip(&a_bc_vec).map(|(l, r)| l - r).collect();
        assoc_sq.push(cd_norm_sq(&assoc_vec));
        ab_c_sq.push(cd_norm_sq(&ab_c_vec));
        a_bc_sq.push(cd_norm_sq(&a_bc_vec));
        cross.push(ab_c_vec.iter().zip(&a_bc_vec).map(|(l, r)| l * r).sum::<f64>());
    }

    let n = n_trials as f64;
    let mean_assoc = assoc_sq.iter().sum::<f64>() / n;
    let mean_ab_c = ab_c_sq.iter().sum::<f64>() / n;
    let mean_a_bc = a_bc_sq.iter().sum::<f64>() / n;
    let mean_cross = cross.iter().sum::<f64>() / n;
    let var_assoc = assoc_sq.iter().map(|x| (x - mean_assoc).powi(2)).sum::<f64>() / n;

    let corr = if mean_ab_c > 0.0 && mean_a_bc > 0.0 {
        mean_cross / (mean_ab_c * mean_a_bc).sqrt()
    } else {
        0.0
    };

    AssociatorStats {
        dim,
        n_trials,
        mean_assoc_sq: mean_assoc,
        std_assoc_sq: var_assoc.sqrt(),
        mean_ab_c_sq: mean_ab_c,
        mean_a_bc_sq: mean_a_bc,
        mean_cross_term: mean_cross,
        correlation_coeff: corr,
    }
}

pub(crate) fn random_unit_vec(dim: usize, rng: &mut impl rand::Rng) -> Vec<f64> {
    use rand_distr::{Distribution, StandardNormal};

    let v: Vec<f64> = (0..dim).map(|_| StandardNormal.sample(rng)).collect();
    let norm = cd_norm_sq(&v).sqrt();
    v.iter().map(|x| x / norm).collect()
}
