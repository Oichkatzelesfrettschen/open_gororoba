//! Subspace basis extraction for Cayley-Dickson algebras.
//!
//! This module provides the two canonical basis-extraction functions that
//! underpin the PMNS mixing angle scan:
//!
//! - [`extract_v6_basis`]: hard-coded dim=16 (sedenion) V_6 subspace, using
//!   nalgebra SVD.  The 6 returned basis vectors span the spectrally-isotropic
//!   complement of the B/C triad column space.
//!
//! - [`extract_vk_basis`]: generalized, dimension-parameterized version using
//!   faer divide-and-conquer eigendecomposition and rayon parallel Gram
//!   accumulation.  The same subspace as `extract_v6_basis` at dim=16,
//!   verified by `test_pathion_vk_spectrum`.
//!
//! # Why these live in flavor_lifts (not algebra_experimental)
//!
//! The extraction logic depends only on `cd_kernel` (sign tables, multiply),
//! `nalgebra`, `faer`, and `rayon` -- all of which are already dependencies
//! of this crate.  Moving here breaks the reverse dependency that previously
//! required `algebra_experimental` to define infrastructure that `flavor_lifts`
//! consumed.

use cd_kernel::cayley_dickson::{SignTable, cd_associator_norm, cd_multiply};
use nalgebra::DMatrix;

/// Extract the V_6 basis from the sedenion (dim=16) incidence matrix algebra.
///
/// The sedenion triad classification yields three types (B, C, X) based on
/// which permutations of the associator [a,b,c] are nonzero. The Type X triads
/// (all three permutations nonzero) span a 27-dimensional column space in
/// assessor coordinates. Projecting out the B/C column space (rank 21) leaves
/// a 6-dimensional complement V_6 that is spectrally isotropic (all singular
/// values equal to 3.420).
///
/// Returns: (6x42 basis matrix, 6 singular values, 42 assessor pairs)
pub fn extract_v6_basis() -> (DMatrix<f64>, Vec<f64>, Vec<(usize, usize)>) {
    let dim = 16_usize;

    // Build assessor index: (low, high) pairs with low in 1..7, high in 9..15,
    // excluding same-offset pairs (high != low + 8)
    let mut assessors: Vec<(usize, usize)> = Vec::new();
    for low in 1..=7_usize {
        for high in 9..=15_usize {
            if high == low + 8 {
                continue;
            }
            assessors.push((low, high));
        }
    }
    assert_eq!(assessors.len(), 42);

    // Build incidence row for a triad (b,c,d): which assessors are touched
    // by the pairwise products e_b*e_c, e_b*e_d, e_c*e_d
    let build_row = |b: usize, c: usize, d: usize| -> Vec<f64> {
        let mut eb = vec![0.0; dim];
        eb[b] = 1.0;
        let mut ec = vec![0.0; dim];
        ec[c] = 1.0;
        let mut ed = vec![0.0; dim];
        ed[d] = 1.0;
        let products = [
            cd_multiply(&eb, &ec),
            cd_multiply(&eb, &ed),
            cd_multiply(&ec, &ed),
        ];
        let mut row = vec![0.0_f64; 42];
        for prod in &products {
            let nonzero: Vec<usize> = prod
                .iter()
                .enumerate()
                .filter(|(_, v)| v.abs() > 1e-12)
                .map(|(i, _)| i)
                .collect();
            if nonzero.len() == 1 {
                let idx = nonzero[0];
                for (a_idx, &(low, high)) in assessors.iter().enumerate() {
                    if idx == low || idx == high {
                        row[a_idx] = 1.0;
                    }
                }
            }
        }
        row
    };

    // Classify all triads into B/C vs X using cd_kernel::cd_associator_norm.
    // assoc_norm(b, c, d) = ||[e_b, e_c, e_d]|| -- nonzero iff non-associative.
    let assoc_norm = |b: usize, c: usize, d: usize| -> f64 {
        let mut eb = vec![0.0; dim];
        eb[b] = 1.0;
        let mut ec = vec![0.0; dim];
        ec[c] = 1.0;
        let mut ed = vec![0.0; dim];
        ed[d] = 1.0;
        cd_associator_norm(&eb, &ec, &ed)
    };

    let mut rows_bc = Vec::new();
    let mut rows_x = Vec::new();

    for b in 1..dim {
        for c in (b + 1)..dim {
            for d in (c + 1)..dim {
                let t1 = assoc_norm(b, c, d);
                let t2 = assoc_norm(b, d, c);
                let t3 = assoc_norm(c, b, d);
                if t1 < 1e-10 && t2 < 1e-10 && t3 < 1e-10 {
                    continue;
                }
                let row = build_row(b, c, d);
                match (t1 > 1e-10, t2 > 1e-10, t3 > 1e-10) {
                    (false, true, false) | (false, false, true) => {
                        rows_bc.push(nalgebra::RowDVector::from_vec(row));
                    }
                    _ => {
                        rows_x.push(nalgebra::RowDVector::from_vec(row));
                    }
                }
            }
        }
    }

    let mat_bc = DMatrix::from_rows(&rows_bc);
    let mat_x = DMatrix::from_rows(&rows_x);

    // SVD of B/C^T to get column space basis
    let svd_bc = mat_bc.transpose().svd(true, false);
    let rank_threshold = 1e-8;
    let u_bc = svd_bc.u.as_ref().unwrap();
    let rank_bc = svd_bc
        .singular_values
        .iter()
        .filter(|&&s| s > rank_threshold)
        .count();

    // Projector: P_BC = Q_BC * Q_BC^T
    let q_bc = u_bc.columns(0, rank_bc);
    let p_bc = q_bc * q_bc.transpose();

    // Residual: C_V6 = X * (I - P_BC)
    let identity = DMatrix::identity(42, 42);
    let proj_complement = &identity - &p_bc;
    let c_v6 = &mat_x * &proj_complement;

    // SVD of C_V6 -> first 6 right singular vectors = V_6 basis
    let svd_v6 = c_v6.svd(false, true);
    let rank_v6 = svd_v6
        .singular_values
        .iter()
        .filter(|&&s| s > rank_threshold)
        .count();

    let vt = svd_v6.v_t.as_ref().unwrap();

    // Extract 6x42 basis matrix (rows = V_6 basis vectors)
    let n_basis = rank_v6.min(6);
    let mut basis_matrix = DMatrix::zeros(n_basis, 42);
    for k in 0..n_basis {
        for col in 0..42 {
            basis_matrix[(k, col)] = vt[(k, col)];
        }
    }

    let singular_values: Vec<f64> = svd_v6
        .singular_values
        .iter()
        .take(n_basis)
        .copied()
        .collect();

    (basis_matrix, singular_values, assessors)
}

// ---------------------------------------------------------------------------
// Private helpers for extract_vk_basis
// ---------------------------------------------------------------------------

/// O(1) associator nonzero check via sign table.
///
/// [a,b,c] = (a*b)*c - a*(b*c) is nonzero iff the left and right sign
/// paths through the CD construction disagree.
fn assoc_nonzero_via_stab(stab: &SignTable, a: usize, b: usize, c: usize) -> bool {
    let sab = stab.sign(a, b);
    let sabc_l = sab * stab.sign(a ^ b, c);
    let sbc = stab.sign(b, c);
    let sabc_r = sbc * stab.sign(a, b ^ c);
    sabc_l != sabc_r
}

/// Symmetrize a faer matrix in-place by averaging each (i,j) and (j,i) pair.
///
/// Returns the maximum asymmetry observed before averaging, which is useful
/// for profiling diagnostics.
fn symmetrize_faer_inplace(m: &mut faer::Mat<f64>) -> f64 {
    let n = m.nrows();
    let mut max_asym = 0.0_f64;
    for i in 0..n {
        for j in (i + 1)..n {
            let asym = (m[(i, j)] - m[(j, i)]).abs();
            if asym > max_asym {
                max_asym = asym;
            }
            let avg = 0.5 * (m[(i, j)] + m[(j, i)]);
            m[(i, j)] = avg;
            m[(j, i)] = avg;
        }
    }
    max_asym
}

/// Emit structural diagnostics for a faer Gram matrix to stderr.
///
/// Caller is responsible for checking `profiling` before calling; this
/// function always writes to stderr.  Separating the guard from the body
/// keeps the outer function's cognitive complexity budget free for
/// structurally important logic.
fn log_gram_diagnostics_to_stderr(name: &str, m: &faer::Mat<f64>) {
    let n = m.nrows();
    let mut max_asym = 0.0_f64;
    let mut nnz_count = 0_usize;
    let mut frob_sq = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            let v = m[(i, j)];
            frob_sq += v * v;
            if v.abs() > 1e-12 {
                nnz_count += 1;
            }
            if j > i {
                let asym = (m[(i, j)] - m[(j, i)]).abs();
                if asym > max_asym {
                    max_asym = asym;
                }
            }
        }
    }
    let total = n * n;
    eprintln!(
        "  [VK_PROFILE] {name}: max_asym_pre={max_asym:.3e}, nnz_fraction={:.4}, frobenius={:.6e}",
        nnz_count as f64 / total as f64,
        frob_sq.sqrt()
    );
}

/// Gram accumulation for a single leading index `b` over all `(c, d)` pairs.
///
/// Each triad `(b, c, d)` with `b < c < d` is classified by which permutations
/// of the associator are nonzero (using the sign-table shortcut), then its
/// XOR-product assessor indices are scattered into `gbc` (B/C-type) or `gx`
/// (X-type).  Counts are accumulated in `count_bc` and `count_x`.
///
/// This is the inner body of the rayon fold; factoring it out eliminates the
/// deep nesting that drives the cognitive-complexity score.
/// Log Gram rayon accumulation counts after Stage 2 parallel fold.
///
/// Called unconditionally; guards on `profiling` internally so the call site
/// avoids an `if profiling` block that would charge the caller's complexity score.
fn log_rayon_gram_counts(
    profiling: bool,
    dim: usize,
    n_assess: usize,
    count_bc: usize,
    count_x: usize,
) {
    if !profiling {
        return;
    }
    eprintln!(
        "  [VK_PROFILE] dim={dim}, n_assess={n_assess}, count_bc={count_bc}, count_x={count_x}"
    );
    eprintln!(
        "  [VK_PROFILE] RAYON_NUM_THREADS={}",
        rayon::current_num_threads()
    );
}

fn accumulate_triad_gram(
    b: usize,
    dim: usize,
    n_assess: usize,
    assess_lookup: &[Vec<usize>],
    stab: &SignTable,
    gram_bufs: (&mut [i64], &mut [i64]),
    counts: (&mut usize, &mut usize),
) {
    let (gbc, gx) = gram_bufs;
    let (count_bc, count_x) = counts;
    let mut nz_buf: Vec<usize> = Vec::with_capacity(8);
    for c in (b + 1)..dim {
        for d in (c + 1)..dim {
            let t1 = assoc_nonzero_via_stab(stab, b, c, d);
            let t2 = assoc_nonzero_via_stab(stab, b, d, c);
            let t3 = assoc_nonzero_via_stab(stab, c, b, d);
            if !t1 && !t2 && !t3 {
                continue;
            }

            nz_buf.clear();
            for &prod_idx in &[b ^ c, b ^ d, c ^ d] {
                if prod_idx > 0 && prod_idx < dim {
                    for &a_idx in &assess_lookup[prod_idx] {
                        if !nz_buf.contains(&a_idx) {
                            nz_buf.push(a_idx);
                        }
                    }
                }
            }

            let is_bc_type = matches!((t1, t2, t3), (false, true, false) | (false, false, true));
            let target: &mut [i64] = if is_bc_type {
                *count_bc += 1;
                &mut *gbc
            } else {
                *count_x += 1;
                &mut *gx
            };
            for &i in &nz_buf {
                for &j in &nz_buf {
                    target[i * n_assess + j] += 1;
                }
            }
        }
    }
}

/// Postprocess the V_k eigendecomposition into a ranked basis matrix.
///
/// Applies a two-level threshold (absolute + relative, with a Frobenius-relative
/// noise guard), sorts retained eigenvectors by descending singular value, and
/// fills the output `(n_basis x n_assess)` basis matrix whose rows are the
/// V_k basis vectors.  Returns `(basis_matrix, singular_values)`.
///
/// # Noise guard rationale
///
/// At dim=64 the complement Gram `G_vk` has `||G_vk||_F = 3.8e-11` but
/// `sv_max = 1.8e-6`.  The ratio `1.8e-6 / 2.75e5 = 6.5e-12` correctly
/// identifies this as pure numerical noise, forcing `rank = 0`.
fn postprocess_vk_eigendecomp(
    vk_eigenvalues: &[f64],
    u_vk: faer::MatRef<'_, f64>,
    gram_x_frob: f64,
    rank_params: (usize, usize),
    thresholds: (f64, f64),
    profiling: bool,
) -> (DMatrix<f64>, Vec<f64>) {
    let (n_assess, max_rank) = rank_params;
    let (abs_eps, rel_eps) = thresholds;
    let vk_sv: Vec<f64> = vk_eigenvalues.iter().map(|&e| e.max(0.0).sqrt()).collect();
    let vk_sv_max = vk_sv.iter().cloned().fold(0.0_f64, f64::max);

    let frob_rel_eps = 1e-8_f64;
    let complement_is_noise = gram_x_frob > 0.0 && vk_sv_max / gram_x_frob < frob_rel_eps;

    let vk_threshold = if complement_is_noise {
        vk_sv_max + 1.0
    } else {
        abs_eps.max(rel_eps * vk_sv_max)
    };

    let mut sv_pairs: Vec<(f64, usize)> = Vec::new();
    for (k, &sv) in vk_sv.iter().enumerate() {
        if sv > vk_threshold {
            sv_pairs.push((sv, k));
        }
    }
    sv_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    let retained_rank_vk = sv_pairs.len();

    if profiling {
        let leading: Vec<f64> = {
            let mut sorted: Vec<f64> = vk_eigenvalues.to_vec();
            sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
            sorted.into_iter().take(5).collect()
        };
        let frob_ratio = if gram_x_frob > 0.0 {
            vk_sv_max / gram_x_frob
        } else {
            0.0
        };
        eprintln!(
            "  [VK_PROFILE] gram_vk: sv_max/||G_x||_F = {frob_ratio:.3e} (noise guard: complement_is_noise={complement_is_noise})"
        );
        eprintln!(
            "  [VK_PROFILE] gram_vk: retained_rank={retained_rank_vk}, threshold={vk_threshold:.3e}"
        );
        eprintln!("  [VK_PROFILE] gram_vk: leading 5 eigenvalues = {leading:?}");
    }

    let n_basis = retained_rank_vk.min(max_rank);

    // basis_matrix shape: (rank, n_assess) -- basis vectors are ROWS
    let mut basis = DMatrix::zeros(n_basis, n_assess);
    for (k, &(_, eig_idx)) in sv_pairs.iter().take(n_basis).enumerate() {
        for col in 0..n_assess {
            basis[(k, col)] = u_vk[(col, eig_idx)];
        }
    }

    let svs: Vec<f64> = sv_pairs
        .iter()
        .take(retained_rank_vk.min(max_rank * 2))
        .map(|&(sv, _)| sv)
        .collect();

    (basis, svs)
}

// ---------------------------------------------------------------------------

/// Generalized V_k basis extraction for any Cayley-Dickson algebra dimension.
///
/// Given a CD algebra of dimension `dim` (must be a power of 2, >= 16), this
/// function constructs the non-associative "assessor complement" subspace --
/// the directions in assessor space that are orthogonal to the B/C (bilinear
/// Cayley) Gram matrix but present in the cross-term Gram matrix.
///
/// # Algorithm (8 stages, instrumented with `VK_PROFILE=1`)
///
/// ```text
///   Stage 1: Build SignTable(dim)              -- O(dim^2) precompute
///   Stage 2: Rayon parallel Gram accumulation  -- C(dim-1, 3) triads
///   Stage 3: i64 -> f64 faer::Mat conversion   -- exact, single pass
///   Stage 4: Eigendecomp gram_bc (faer D&C)    -- selfadjoint_eigendecomposition
///   Stage 5: Projector P_BC from retained eigenvectors
///   Stage 6: Complement matmul: G_vk = P_perp * G_x * P_perp
///   Stage 7: Eigendecomp gram_vk (faer D&C)    -- selfadjoint_eigendecomposition
///   Stage 8: Threshold + descending sort + basis extraction
/// ```
///
/// # Assessor geometry
///
/// ```text
///   dim=16 (sedenion):  low in 1..7,  high in 9..15   -> 42 assessor pairs
///   dim=32 (Pathion):   low in 1..15, high in 17..31  -> 210 assessor pairs
///   dim=64 (Chingon):   low in 1..31, high in 33..63  -> 930 assessor pairs
/// ```
///
/// Same-offset pairs (`high == low + half`) are excluded -- they are the
/// "doubled identity" directions that carry no non-associative information.
///
/// # Rank threshold
///
/// Two-level: absolute+relative with a Frobenius-relative noise guard.
/// See `postprocess_vk_eigendecomp` for full rationale.
///
/// # Profiling
///
/// Set `VK_PROFILE=1` to emit per-stage wall-clock timing and structural
/// diagnostics to stderr.
pub fn extract_vk_basis(
    dim: usize,
    max_rank: usize,
) -> (DMatrix<f64>, Vec<f64>, Vec<(usize, usize)>) {
    assert!(
        dim.is_power_of_two() && dim >= 16,
        "dim must be power of 2 >= 16"
    );
    let half = dim / 2;
    let profiling = std::env::var("VK_PROFILE").is_ok();

    macro_rules! profile_stage {
        ($stage:expr, $name:expr, $body:expr) => {{
            let _t0 = std::time::Instant::now();
            let result = $body;
            if profiling {
                eprintln!(
                    "  [VK_PROFILE] Stage {}: {} -- {:.3}s",
                    $stage,
                    $name,
                    _t0.elapsed().as_secs_f64()
                );
            }
            result
        }};
    }

    // Stage 1: Sign table construction
    let stab = profile_stage!(1, "sign table construction", { SignTable::new(dim) });

    // Build assessor index: (low, high) with low in 1..half-1, high in half+1..dim-1,
    // excluding same-offset pairs (high == low + half)
    let mut assessors: Vec<(usize, usize)> = Vec::new();
    for low in 1..half {
        for high in (half + 1)..dim {
            if high == low + half {
                continue;
            }
            assessors.push((low, high));
        }
    }
    let n_assess = assessors.len();

    // Build assessor lookup: idx -> position in assessors vec.
    let mut assess_lookup = vec![Vec::new(); dim];
    for (a_idx, &(low, high)) in assessors.iter().enumerate() {
        assess_lookup[low].push(a_idx);
        assess_lookup[high].push(a_idx);
    }

    // Stage 2: Rayon parallel Gram accumulation (triple loop).
    // Integer accumulation gives bit-identical results regardless of thread scheduling.
    // The inner body is factored into accumulate_triad_gram to reduce nesting depth.
    use rayon::prelude::*;
    let nn = n_assess * n_assess;
    let (gram_bc_flat, gram_x_flat, count_bc, count_x) =
        profile_stage!(2, "rayon Gram accumulation", {
            (1..dim)
                .into_par_iter()
                .fold(
                    || (vec![0i64; nn], vec![0i64; nn], 0usize, 0usize),
                    |(mut gbc, mut gx, mut cbc, mut cx), b| {
                        accumulate_triad_gram(
                            b,
                            dim,
                            n_assess,
                            &assess_lookup,
                            &stab,
                            (&mut gbc, &mut gx),
                            (&mut cbc, &mut cx),
                        );
                        (gbc, gx, cbc, cx)
                    },
                )
                .reduce(
                    || (vec![0i64; nn], vec![0i64; nn], 0, 0),
                    |(mut a0, mut a1, a2, a3), (b0, b1, b2, b3)| {
                        for i in 0..nn {
                            a0[i] += b0[i];
                        }
                        for i in 0..nn {
                            a1[i] += b1[i];
                        }
                        (a0, a1, a2 + b2, a3 + b3)
                    },
                )
        });

    log_rayon_gram_counts(profiling, dim, n_assess, count_bc, count_x);

    if count_bc == 0 || count_x == 0 {
        return (DMatrix::zeros(0, n_assess), vec![], assessors);
    }

    // Stage 3: i64 -> f64 faer::Mat conversion
    let (gram_bc_faer, gram_x_faer, gram_x_frob) =
        profile_stage!(3, "i64 -> f64 faer::Mat conversion", {
            let mut gbc = faer::Mat::<f64>::zeros(n_assess, n_assess);
            let mut gx = faer::Mat::<f64>::zeros(n_assess, n_assess);
            let mut gx_frob_sq = 0.0_f64;
            for i in 0..n_assess {
                for j in 0..n_assess {
                    gbc[(i, j)] = gram_bc_flat[i * n_assess + j] as f64;
                    let v = gram_x_flat[i * n_assess + j] as f64;
                    gx[(i, j)] = v;
                    gx_frob_sq += v * v;
                }
            }
            (gbc, gx, gx_frob_sq.sqrt())
        });

    // Stage 4: First eigendecomposition (gram_bc) -- identifies B/C column space
    if profiling {
        log_gram_diagnostics_to_stderr("gram_bc", &gram_bc_faer);
    }
    let mut gram_bc_sym = gram_bc_faer;
    let asym_bc = symmetrize_faer_inplace(&mut gram_bc_sym);
    if profiling {
        eprintln!("  [VK_PROFILE] gram_bc: max_asym_post symmetrize = {asym_bc:.3e}");
    }
    let eig_bc = profile_stage!(4, "eigendecomp gram_bc (faer)", {
        gram_bc_sym.self_adjoint_eigen(faer::Side::Lower).unwrap()
    });

    let abs_eps = 1e-6_f64;
    let rel_eps = 1e-4_f64;

    let bc_eigenvalues: Vec<f64> = eig_bc.S().column_vector().iter().copied().collect();
    let bc_sv: Vec<f64> = bc_eigenvalues.iter().map(|&e| e.max(0.0).sqrt()).collect();
    let bc_sv_max = bc_sv.iter().cloned().fold(0.0_f64, f64::max);
    let bc_threshold = abs_eps.max(rel_eps * bc_sv_max);
    let retained_rank_bc = bc_sv.iter().filter(|&&s| s > bc_threshold).count();

    if profiling {
        let leading: Vec<f64> = {
            let mut sorted = bc_eigenvalues.clone();
            sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
            sorted.into_iter().take(5).collect()
        };
        eprintln!(
            "  [VK_PROFILE] gram_bc: retained_rank={retained_rank_bc}, threshold={bc_threshold:.3e}"
        );
        eprintln!("  [VK_PROFILE] gram_bc: leading 5 eigenvalues = {leading:?}");
    }

    // Stage 5: Projector P_BC = sum_k |v_k><v_k| for retained eigenvectors
    let p_bc = profile_stage!(5, "projector construction P_BC", {
        let u_bc = eig_bc.U();
        let mut p = DMatrix::zeros(n_assess, n_assess);
        for (k, &sv) in bc_sv.iter().enumerate() {
            if sv > bc_threshold {
                let mut col = nalgebra::DVector::zeros(n_assess);
                for i in 0..n_assess {
                    col[i] = u_bc[(i, k)];
                }
                p += &col * col.transpose();
            }
        }
        p
    });

    // Stage 6: Complement matmul -- G_vk = P_perp * G_x * P_perp
    let gram_vk_faer = profile_stage!(6, "complement matmul P_perp*G_x*P_perp", {
        let identity = DMatrix::identity(n_assess, n_assess);
        let proj_complement = &identity - &p_bc;
        let p_perp_faer = faer::Mat::from_fn(n_assess, n_assess, |r, c| proj_complement[(r, c)]);
        &p_perp_faer * &gram_x_faer * &p_perp_faer
    });

    // Stage 7: Second eigendecomposition (gram_vk) -- extracts V_k basis
    if profiling {
        log_gram_diagnostics_to_stderr("gram_vk", &gram_vk_faer);
    }
    let mut gram_vk_sym = gram_vk_faer;
    let asym_vk = symmetrize_faer_inplace(&mut gram_vk_sym);
    if profiling {
        eprintln!("  [VK_PROFILE] gram_vk: max_asym_post symmetrize = {asym_vk:.3e}");
    }
    let eig_vk = profile_stage!(7, "eigendecomp gram_vk (faer)", {
        gram_vk_sym.self_adjoint_eigen(faer::Side::Lower).unwrap()
    });

    // Stage 8: Postprocessing -- threshold, canonical sort, basis extraction.
    // Factored into postprocess_vk_eigendecomp to keep nesting depth flat here.
    let (basis_matrix, singular_values) =
        profile_stage!(8, "postprocessing (sort, extract basis)", {
            let vk_evals: Vec<f64> = eig_vk.S().column_vector().iter().copied().collect();
            let u_vk = eig_vk.U();
            postprocess_vk_eigendecomp(
                &vk_evals,
                u_vk,
                gram_x_frob,
                (n_assess, max_rank),
                (abs_eps, rel_eps),
                profiling,
            )
        });

    (basis_matrix, singular_values, assessors)
}
