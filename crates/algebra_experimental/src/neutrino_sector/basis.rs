// Graduated to flavor_lifts::basis.
// Re-exported here so that all existing call sites in neutrino_sector::tests::*
// continue to resolve without any change.
pub use flavor_lifts::basis::{extract_v6_basis, extract_vk_basis};

/// nalgebra-based eigendecomp fallback, preserved for backend regression.
///
/// This is the pre-faer implementation of [`extract_vk_basis`], using nalgebra's
/// `symmetric_eigen()` (Jacobi rotations) instead of faer's divide-and-conquer.
/// It exists solely so that `test_faer_vs_nalgebra_eigendecomp` can verify
/// that the faer migration did not change the extracted subspace.
///
/// # Differences from the production version
///
/// - Uses nalgebra `symmetric_eigen()` (Jacobi) instead of faer D&C
/// - Uses the old pure-relative threshold `sigma_threshold = 1e-4`
///   (squared to `eig_threshold = 1e-8` for eigenvalue comparison)
/// - No profiling instrumentation, no symmetrization, no diagnostics
///
/// # When to remove
///
/// After the faer backend has been validated at dim=64 and the timing
/// results recorded, this function can be deleted.  The projector
/// agreement test is the single gate for removal.
#[cfg(test)]
pub(crate) fn extract_vk_basis_nalgebra(
    dim: usize, max_rank: usize,
) -> (nalgebra::DMatrix<f64>, Vec<f64>, Vec<(usize, usize)>) {
    use cd_kernel::cayley_dickson::SignTable;
    use nalgebra::DMatrix;

    assert!(dim.is_power_of_two() && dim >= 16);
    let half = dim / 2;
    let stab = SignTable::new(dim);

    let mut assessors: Vec<(usize, usize)> = Vec::new();
    for low in 1..half {
        for high in (half + 1)..dim {
            if high == low + half { continue; }
            assessors.push((low, high));
        }
    }
    let n_assess = assessors.len();
    let mut assess_lookup = vec![Vec::new(); dim];
    for (a_idx, &(low, high)) in assessors.iter().enumerate() {
        assess_lookup[low].push(a_idx);
        assess_lookup[high].push(a_idx);
    }

    let assoc_nonzero = |a: usize, b: usize, c: usize| -> bool {
        let sab = stab.sign(a, b);
        let sabc_l = sab * stab.sign(a ^ b, c);
        let sbc = stab.sign(b, c);
        let sabc_r = sbc * stab.sign(a, b ^ c);
        sabc_l != sabc_r
    };

    use rayon::prelude::*;
    let nn = n_assess * n_assess;
    let (gram_bc_flat, gram_x_flat, count_bc, count_x) = (1..dim)
        .into_par_iter()
        .fold(
            || (vec![0i64; nn], vec![0i64; nn], 0usize, 0usize),
            |(mut gbc, mut gx, mut cbc, mut cx), b| {
                let mut nz_buf: Vec<usize> = Vec::with_capacity(8);
                for c in (b + 1)..dim {
                    for d in (c + 1)..dim {
                        let t1 = assoc_nonzero(b, c, d);
                        let t2 = assoc_nonzero(b, d, c);
                        let t3 = assoc_nonzero(c, b, d);
                        if !t1 && !t2 && !t3 { continue; }
                        nz_buf.clear();
                        for &prod_idx in &[b ^ c, b ^ d, c ^ d] {
                            if prod_idx > 0 && prod_idx < dim {
                                for &a_idx in &assess_lookup[prod_idx] {
                                    if !nz_buf.contains(&a_idx) { nz_buf.push(a_idx); }
                                }
                            }
                        }
                        let target = match (t1, t2, t3) {
                            (false, true, false) | (false, false, true) => { cbc += 1; &mut gbc }
                            _ => { cx += 1; &mut gx }
                        };
                        for &i in &nz_buf { for &j in &nz_buf { target[i * n_assess + j] += 1; } }
                    }
                }
                (gbc, gx, cbc, cx)
            },
        )
        .reduce(
            || (vec![0i64; nn], vec![0i64; nn], 0, 0),
            |(mut a0, mut a1, a2, a3), (b0, b1, b2, b3)| {
                for i in 0..nn { a0[i] += b0[i]; }
                for i in 0..nn { a1[i] += b1[i]; }
                (a0, a1, a2 + b2, a3 + b3)
            },
        );

    if count_bc == 0 || count_x == 0 {
        return (DMatrix::zeros(0, n_assess), vec![], assessors);
    }

    let mut gram_bc = DMatrix::zeros(n_assess, n_assess);
    let mut gram_x = DMatrix::zeros(n_assess, n_assess);
    for i in 0..n_assess {
        for j in 0..n_assess {
            gram_bc[(i, j)] = gram_bc_flat[i * n_assess + j] as f64;
            gram_x[(i, j)] = gram_x_flat[i * n_assess + j] as f64;
        }
    }

    let sigma_threshold = 1e-4;
    let eig_threshold = sigma_threshold * sigma_threshold;
    let eig_bc = gram_bc.symmetric_eigen();

    let mut p_bc = DMatrix::zeros(n_assess, n_assess);
    for k in 0..n_assess {
        if eig_bc.eigenvalues[k] > eig_threshold {
            let col = eig_bc.eigenvectors.column(k);
            p_bc += col * col.transpose();
        }
    }

    let identity = DMatrix::identity(n_assess, n_assess);
    let proj_complement = &identity - &p_bc;
    let gram_vk = &proj_complement * &gram_x * &proj_complement;
    let eig_vk = gram_vk.symmetric_eigen();

    let mut sv_pairs: Vec<(f64, usize)> = Vec::new();
    for k in 0..n_assess {
        let ev: f64 = eig_vk.eigenvalues[k];
        if ev > eig_threshold {
            sv_pairs.push((ev.sqrt(), k));
        }
    }
    sv_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    let rank_vk = sv_pairs.len();
    let n_basis = rank_vk.min(max_rank);
    let mut basis_matrix = DMatrix::zeros(n_basis, n_assess);
    for (k, &(_, eig_idx)) in sv_pairs.iter().take(n_basis).enumerate() {
        for col in 0..n_assess {
            basis_matrix[(k, col)] = eig_vk.eigenvectors[(col, eig_idx)];
        }
    }

    let singular_values: Vec<f64> = sv_pairs.iter()
        .take(rank_vk.min(max_rank * 2))
        .map(|&(sv, _)| sv)
        .collect();

    (basis_matrix, singular_values, assessors)
}
