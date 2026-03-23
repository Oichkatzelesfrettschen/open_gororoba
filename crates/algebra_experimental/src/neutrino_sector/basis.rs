/// Extract V_6 basis from incidence matrix algebra.
///
/// The sedenion triad classification yields three types (B, C, X) based on
/// which permutations of the associator [a,b,c] are nonzero. The Type X triads
/// (all three permutations nonzero) span a 27-dimensional column space in
/// assessor coordinates. Projecting out the B/C column space (rank 21) leaves
/// a 6-dimensional complement V_6 that is spectrally isotropic (all singular
/// values equal to 3.420).
///
/// Returns: (6x42 basis matrix, 6 singular values, 42 assessor pairs)
pub fn extract_v6_basis() -> (nalgebra::DMatrix<f64>, Vec<f64>, Vec<(usize, usize)>) {
    use cd_kernel::cayley_dickson::cd_multiply;
    use crate::sedenion_subalgebras::assoc_strict;
    use nalgebra::DMatrix;

    let dim = 16_usize;

    // Build assessor index: (low, high) pairs with low in 1..7, high in 9..15,
    // excluding same-offset pairs (high != low + 8)
    let mut assessors: Vec<(usize, usize)> = Vec::new();
    for low in 1..=7_usize {
        for high in 9..=15_usize {
            if high == low + 8 { continue; }
            assessors.push((low, high));
        }
    }
    assert_eq!(assessors.len(), 42);

    // Build incidence row for a triad (b,c,d): which assessors are touched
    // by the pairwise products e_b*e_c, e_b*e_d, e_c*e_d
    let build_row = |b: usize, c: usize, d: usize| -> Vec<f64> {
        let mut eb = vec![0.0; dim]; eb[b] = 1.0;
        let mut ec = vec![0.0; dim]; ec[c] = 1.0;
        let mut ed = vec![0.0; dim]; ed[d] = 1.0;
        let products = [
            cd_multiply(&eb, &ec),
            cd_multiply(&eb, &ed),
            cd_multiply(&ec, &ed),
        ];
        let mut row = vec![0.0_f64; 42];
        for prod in &products {
            let nonzero: Vec<usize> = prod.iter().enumerate()
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

    // Classify all triads into B/C vs X
    let mut rows_bc = Vec::new();
    let mut rows_x = Vec::new();

    for b in 1..dim {
        for c in (b + 1)..dim {
            for d in (c + 1)..dim {
                let t1 = assoc_strict(dim, b, c, d);
                let t2 = assoc_strict(dim, b, d, c);
                let t3 = assoc_strict(dim, c, b, d);
                if t1 < 1e-10 && t2 < 1e-10 && t3 < 1e-10 { continue; }
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
    let rank_bc = svd_bc.singular_values.iter()
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
    let rank_v6 = svd_v6.singular_values.iter()
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

    let singular_values: Vec<f64> = svd_v6.singular_values.iter()
        .take(n_basis)
        .copied()
        .collect();

    (basis_matrix, singular_values, assessors)
}

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
/// # Eigendecomp backend
///
/// Uses faer `selfadjoint_eigendecomposition(Side::Lower)` (divide-and-conquer)
/// rather than nalgebra `symmetric_eigen()` (Jacobi rotations). D&C has better
/// cache behavior and parallelizes naturally on the 930x930 matrices at dim=64.
/// The nalgebra fallback is preserved as `extract_vk_basis_nalgebra` behind
/// `#[cfg(test)]` for backend regression.
///
/// # Rank threshold
///
/// Two-level: absolute+relative with a Frobenius-relative noise guard.
///
/// Level 1: `threshold = max(1e-6, 1e-4 * sv_max)` handles small matrices
/// where the leading SV is a genuine signal.
///
/// Level 2: if `sv_max / ||G_x||_F < 1e-8`, the entire complement matrix
/// is numerical noise and rank is forced to 0.  This catches the dim=64
/// case where `||G_vk||_F = 3.8e-11` but `sv_max = 1.8e-6` (just above
/// the absolute floor).  The Frobenius ratio `1.8e-6 / 2.75e5 = 6.5e-12`
/// correctly identifies this as noise.
///
/// Tiny negative eigenvalues from PSD violation are clamped via
/// `e.max(0.0).sqrt()` before thresholding.
///
/// # Invariants
///
/// ```text
///   basis_matrix shape : (rank, n_assess)  -- basis vectors are ROWS
///   orthonormality     : |B * B^T - I_rank|_F < 1e-10
///   projector          : P = B^T * B is the n_assess x n_assess projector
///   rank pattern       : dim=16 -> 6, dim=32 -> 1, dim=64 -> 0
///                        (Frobenius-relative noise guard: sv_max/||G_x||_F < 1e-8)
/// ```
///
/// # Profiling
///
/// Set `VK_PROFILE=1` to emit per-stage wall-clock timing and structural
/// diagnostics (max asymmetry, nnz fraction, Frobenius norm, retained rank,
/// leading 5 eigenvalues) to stderr.
///
/// # Callers
///
/// - `test_numerical_regression_baselines`: rank=6 and sv degeneracy at dim=16
/// - `test_pathion_vk_spectrum`: rank pattern across dim=16/32/64
/// - `test_faer_vs_nalgebra_eigendecomp`: backend subspace agreement
///
/// # Why this is a separate function from [`extract_v6_basis`]
///
/// `extract_v6_basis` is hard-coded for dim=16 with the original nalgebra
/// backend and the "6" baked into variable names.  This generalized version
/// parameterizes by dimension and uses the faster faer backend, but both
/// compute the same subspace for dim=16 (verified by `test_pathion_vk_spectrum`).
pub fn extract_vk_basis(dim: usize, max_rank: usize) -> (nalgebra::DMatrix<f64>, Vec<f64>, Vec<(usize, usize)>) {
    use cd_kernel::cayley_dickson::SignTable;
    use nalgebra::DMatrix;

    assert!(dim.is_power_of_two() && dim >= 16, "dim must be power of 2 >= 16");
    let half = dim / 2;
    let profiling = std::env::var("VK_PROFILE").is_ok();

    macro_rules! profile_stage {
        ($stage:expr, $name:expr, $body:expr) => {{
            let _t0 = std::time::Instant::now();
            let result = $body;
            if profiling {
                eprintln!("  [VK_PROFILE] Stage {}: {} -- {:.3}s",
                    $stage, $name, _t0.elapsed().as_secs_f64());
            }
            result
        }};
    }

    // Stage 1: Sign table construction
    let stab = profile_stage!(1, "sign table construction", {
        SignTable::new(dim)
    });

    // Build assessor index: (low, high) with low in 1..half-1, high in half+1..dim-1,
    // excluding same-offset pairs (high == low + half)
    let mut assessors: Vec<(usize, usize)> = Vec::new();
    for low in 1..half {
        for high in (half + 1)..dim {
            if high == low + half { continue; }
            assessors.push((low, high));
        }
    }
    let n_assess = assessors.len();

    // Build assessor lookup: idx -> position in assessors vec.
    // For a product e_p * e_q = +/- e_{p^q}, the result index is p^q.
    // We need to check if p^q matches any assessor's low or high component.
    let mut assess_lookup = vec![Vec::new(); dim];
    for (a_idx, &(low, high)) in assessors.iter().enumerate() {
        assess_lookup[low].push(a_idx);
        assess_lookup[high].push(a_idx);
    }

    // O(1) associator check via sign table:
    // [a,b,c] = (a*b)*c - a*(b*c)
    // e_a*e_b = s(a,b)*e_{a^b}, then (a*b)*c = s(a,b)*s(a^b,c)*e_{a^b^c}
    // a*(b*c) = s(b,c)*s(a,b^c)*e_{a^b^c}
    // [a,b,c] = (s(a,b)*s(a^b,c) - s(b,c)*s(a,b^c)) * e_{a^b^c}
    // nonzero iff s(a,b)*s(a^b,c) != s(b,c)*s(a,b^c)
    let assoc_nonzero = |a: usize, b: usize, c: usize| -> bool {
        let sab = stab.sign(a, b);
        let sabc_l = sab * stab.sign(a ^ b, c);
        let sbc = stab.sign(b, c);
        let sabc_r = sbc * stab.sign(a, b ^ c);
        sabc_l != sabc_r
    };

    // Accumulate Gram matrices G_bc = M_bc^T * M_bc and G_x = M_x^T * M_x
    // via rayon-parallel fold+reduce with integer accumulation.
    //
    // # Why integer accumulation
    //
    // The Gram updates are `+= 1` (binary incidence rows). Accumulating as
    // i64 in thread-local flat buffers avoids the non-determinism of parallel
    // f64 addition (rayon does not guarantee fold/reduce ordering). The final
    // i64-to-f64 conversion happens ONCE after the reduce, giving bit-identical
    // results to the sequential path.
    //
    // # Why rayon here
    //
    // For dim=64: C(63,3)=39,711 triads, each with 3 sign-table lookups and
    // ~6 sparse Gram updates. Sequential: ~240s. With 8-core rayon fold+reduce
    // on the outer b-loop: expected ~30-60s.
    use rayon::prelude::*;

    // Stage 2: Rayon Gram accumulation (triple loop)
    let nn = n_assess * n_assess;
    let (gram_bc_flat, gram_x_flat, count_bc, count_x) = profile_stage!(2, "rayon Gram accumulation", {
        (1..dim)
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
                                        if !nz_buf.contains(&a_idx) {
                                            nz_buf.push(a_idx);
                                        }
                                    }
                                }
                            }

                            let target = match (t1, t2, t3) {
                                (false, true, false) | (false, false, true) => {
                                    cbc += 1;
                                    &mut gbc
                                }
                                _ => {
                                    cx += 1;
                                    &mut gx
                                }
                            };
                            for &i in &nz_buf {
                                for &j in &nz_buf {
                                    target[i * n_assess + j] += 1;
                                }
                            }
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
            )
    });

    if profiling {
        eprintln!("  [VK_PROFILE] dim={dim}, n_assess={n_assess}, count_bc={count_bc}, count_x={count_x}");
        eprintln!("  [VK_PROFILE] RAYON_NUM_THREADS={}", rayon::current_num_threads());
    }

    if count_bc == 0 || count_x == 0 {
        return (DMatrix::zeros(0, n_assess), vec![], assessors);
    }

    // ---------------------------------------------------------------------------
    // Stage 3: i64 -> f64 faer::Mat conversion
    // ---------------------------------------------------------------------------
    // Build faer matrices directly from the flat i64 buffers.  We skip the
    // nalgebra DMatrix intermediate that the old code used -- the only consumer
    // of the Gram matrices is now faer's eigendecomposition.
    let (gram_bc_faer, gram_x_faer, gram_x_frob) = profile_stage!(3, "i64 -> f64 faer::Mat conversion", {
        let mut gbc = faer::Mat::<f64>::zeros(n_assess, n_assess);
        let mut gx = faer::Mat::<f64>::zeros(n_assess, n_assess);
        let mut gx_frob_sq = 0.0_f64;
        for i in 0..n_assess {
            for j in 0..n_assess {
                gbc.write(i, j, gram_bc_flat[i * n_assess + j] as f64);
                let v = gram_x_flat[i * n_assess + j] as f64;
                gx.write(i, j, v);
                gx_frob_sq += v * v;
            }
        }
        (gbc, gx, gx_frob_sq.sqrt())
    });

    // ---------------------------------------------------------------------------
    // Structural diagnostics (profiling only)
    // ---------------------------------------------------------------------------
    // These two closures instrument the eigendecomp inputs.  They answer the
    // question "is my matrix actually symmetric and dense, or am I paying for
    // structure I don't have?"  The answer at dim=16 is: perfectly symmetric
    // (integer construction), nnz=1.0, dense.  At dim=64 the projector matmul
    // introduces ~1e-14 asymmetry which the symmetrize step erases.

    // Log max |M[i,j]-M[j,i]|, fraction of entries > 1e-12, and ||M||_F.
    let log_gram_diagnostics = |name: &str, m: &faer::Mat<f64>| {
        if !profiling { return; }
        let n = m.nrows();
        let mut max_asym = 0.0_f64;
        let mut nnz_count = 0_usize;
        let mut frob_sq = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                let v = m.read(i, j);
                frob_sq += v * v;
                if v.abs() > 1e-12 { nnz_count += 1; }
                if j > i {
                    let asym = (m.read(i, j) - m.read(j, i)).abs();
                    if asym > max_asym { max_asym = asym; }
                }
            }
        }
        let total = n * n;
        eprintln!("  [VK_PROFILE] {name}: max_asym_pre={max_asym:.3e}, nnz_fraction={:.4}, frobenius={:.6e}",
            nnz_count as f64 / total as f64, frob_sq.sqrt());
    };

    // Force exact symmetry: M[i,j] = M[j,i] = avg(M[i,j], M[j,i]).
    // Returns the max defect BEFORE overwriting, so the caller can log it.
    // Why symmetrize at all?  faer's selfadjoint_eigendecomposition reads
    // only the lower triangle (Side::Lower), but a defect > ~1e-12 in the
    // upper triangle would cause nalgebra's Jacobi solver to give different
    // results, making backend comparison unreliable.
    let symmetrize = |m: &mut faer::Mat<f64>| -> f64 {
        let n = m.nrows();
        let mut max_asym = 0.0_f64;
        for i in 0..n {
            for j in (i + 1)..n {
                let asym = (m.read(i, j) - m.read(j, i)).abs();
                if asym > max_asym { max_asym = asym; }
                let avg = 0.5 * (m.read(i, j) + m.read(j, i));
                m.write(i, j, avg);
                m.write(j, i, avg);
            }
        }
        max_asym
    };

    // ---------------------------------------------------------------------------
    // Stage 4: First eigendecomposition (gram_bc)
    // ---------------------------------------------------------------------------
    // This gives us the column space of the B/C (bilinear Cayley) triad
    // incidence matrix.  We need the eigenvectors to build the projector P_BC
    // that will be subtracted off in Stage 6.
    //
    // Why faer instead of nalgebra?  faer uses divide-and-conquer after
    // Householder tridiagonalization, which has better cache locality than
    // nalgebra's Jacobi rotations for n > ~100.  At n=930 (dim=64), this
    // is the dominant cost center.
    log_gram_diagnostics("gram_bc", &gram_bc_faer);
    let mut gram_bc_sym = gram_bc_faer;
    let asym_bc = symmetrize(&mut gram_bc_sym);
    if profiling {
        eprintln!("  [VK_PROFILE] gram_bc: max_asym_post symmetrize = {asym_bc:.3e}");
    }
    let eig_bc = profile_stage!(4, "eigendecomp gram_bc (faer)", {
        gram_bc_sym.selfadjoint_eigendecomposition(faer::Side::Lower)
    });

    // ---------------------------------------------------------------------------
    // Rank thresholding (two-level, shared between Stages 4 and 7)
    // ---------------------------------------------------------------------------
    // Level 1: threshold = max(abs_eps, rel_eps * sv_max)
    //   Handles the per-eigenvalue decision for matrices with genuine signal.
    //
    // Level 2 (gram_vk only): Frobenius-relative noise guard.
    //   If sv_max / ||G_x||_F < 1e-8, the entire complement is noise and
    //   rank is forced to 0.  This catches the dim=64 case where
    //   ||G_vk||_F = 3.8e-11, sv_max = 1.8e-6, but the ratio to the input
    //   Gram (||G_x||_F = 2.5e5) is only 7.5e-12 -- pure numerical noise.
    //
    // Why clamp negatives?  The Gram matrices are PSD by construction
    // (G = M^T M), but floating-point accumulation can produce eigenvalues
    // as negative as -1e-14.  Clamping via e.max(0.0) before sqrt avoids
    // NaN and correctly treats these as zero-rank directions.
    let abs_eps = 1e-6_f64;
    let rel_eps = 1e-4_f64;

    let bc_eigenvalues: Vec<f64> = (0..n_assess)
        .map(|k| eig_bc.s().column_vector().read(k))
        .collect();
    // Singular values = sqrt(max(eigenvalue, 0))
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
        eprintln!("  [VK_PROFILE] gram_bc: retained_rank={retained_rank_bc}, threshold={bc_threshold:.3e}");
        eprintln!("  [VK_PROFILE] gram_bc: leading 5 eigenvalues = {leading:?}");
    }

    // ---------------------------------------------------------------------------
    // Stage 5: Projector P_BC = sum_k |v_k><v_k| for retained eigenvectors
    // ---------------------------------------------------------------------------
    // P_BC projects onto the column space of the B/C Gram matrix.  The
    // complement (I - P_BC) will isolate the "V_k" directions in Stage 6.
    //
    // Why build in nalgebra?  The outer-product accumulation p += v*v^T is
    // a dense rank-1 update.  nalgebra's column-major DMatrix handles this
    // naturally.  We convert back to faer only for the Stage 6 matmul.
    let p_bc = profile_stage!(5, "projector construction P_BC", {
        let u_bc = eig_bc.u();
        let mut p = DMatrix::zeros(n_assess, n_assess);
        for (k, &sv) in bc_sv.iter().enumerate() {
            if sv > bc_threshold {
                let mut col = nalgebra::DVector::zeros(n_assess);
                for i in 0..n_assess {
                    col[i] = u_bc.read(i, k);
                }
                p += &col * col.transpose();
            }
        }
        p
    });

    // ---------------------------------------------------------------------------
    // Stage 6: Complement matmul -- G_vk = P_perp * G_x * P_perp
    // ---------------------------------------------------------------------------
    // P_perp = I - P_BC.  The triple product isolates the Gram structure of
    // the "cross" triads (those not in the B/C family) projected onto the
    // orthogonal complement of the B/C column space.
    //
    // Decision gate 8e: if this stage dominates after the faer swap, the
    // low-rank trick (cost O(r*n^2) instead of O(n^3)) should be applied:
    //   P_perp * X * P_perp = X - B^T(BX) - (XB^T)B + B^T(BXB^T)B
    // where B is the (r x n) retained basis and r << n.
    let gram_vk_faer = profile_stage!(6, "complement matmul P_perp*G_x*P_perp", {
        let identity = DMatrix::identity(n_assess, n_assess);
        let proj_complement = &identity - &p_bc;
        let p_perp_faer = faer::Mat::from_fn(n_assess, n_assess, |r, c| proj_complement[(r, c)]);
        &p_perp_faer * &gram_x_faer * &p_perp_faer
    });

    // ---------------------------------------------------------------------------
    // Stage 7: Second eigendecomposition (gram_vk)
    // ---------------------------------------------------------------------------
    // This is the one that actually extracts the V_k basis.  The spectrum
    // of gram_vk reveals the non-associative directions that are independent
    // of the B/C structure.
    //
    // For dim=16: 6 nonzero eigenvalues (all degenerate at ~11.696), giving
    // the V_6 subspace that steers three PMNS mixing angles + three masses.
    // For dim=32: 1 nonzero eigenvalue.  For dim=64: 0 (rank drops to zero).
    log_gram_diagnostics("gram_vk", &gram_vk_faer);
    let mut gram_vk_sym = gram_vk_faer;
    let asym_vk = symmetrize(&mut gram_vk_sym);
    if profiling {
        eprintln!("  [VK_PROFILE] gram_vk: max_asym_post symmetrize = {asym_vk:.3e}");
    }
    let eig_vk = profile_stage!(7, "eigendecomp gram_vk (faer)", {
        gram_vk_sym.selfadjoint_eigendecomposition(faer::Side::Lower)
    });

    // ---------------------------------------------------------------------------
    // Stage 8: Postprocessing -- threshold, canonical sort, basis extraction
    // ---------------------------------------------------------------------------
    let (basis_matrix, singular_values) = profile_stage!(8, "postprocessing (sort, extract basis)", {
        // faer returns eigenvalues in nondecreasing order.  We want descending
        // singular values for a canonical, backend-independent output.
        let vk_eigenvalues: Vec<f64> = (0..n_assess)
            .map(|k| eig_vk.s().column_vector().read(k))
            .collect();
        // Clamp negatives to zero (see threshold rationale above)
        let vk_sv: Vec<f64> = vk_eigenvalues.iter().map(|&e| e.max(0.0).sqrt()).collect();
        let vk_sv_max = vk_sv.iter().cloned().fold(0.0_f64, f64::max);

        // Frobenius-relative guard: if the leading SV of gram_vk is
        // negligible relative to the input Gram's Frobenius norm, the
        // entire complement matrix is numerical noise and rank is 0.
        //
        // Why this is needed: at dim=64, gram_vk has ||G_vk||_F = 3.8e-11
        // but sv_max = sqrt(3.4e-12) = 1.8e-6.  Without this guard, the
        // absolute floor abs_eps = 1e-6 lets noise through (1.8e-6 > 1e-6).
        // The Frobenius guard detects that 1.8e-6 / 2.75e5 = 6.5e-12,
        // far below any physical signal.
        let frob_rel_eps = 1e-8_f64;
        let complement_is_noise = gram_x_frob > 0.0
            && vk_sv_max / gram_x_frob < frob_rel_eps;

        let vk_threshold = if complement_is_noise {
            // Everything is noise -- set threshold above sv_max to get rank=0
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
        // Single canonical descending sort.  (The old code had a duplicate sort
        // at this point -- removed in the faer migration.)
        sv_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        let retained_rank_vk = sv_pairs.len();

        if profiling {
            let leading: Vec<f64> = {
                let mut sorted = vk_eigenvalues.clone();
                sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
                sorted.into_iter().take(5).collect()
            };
            let frob_ratio = if gram_x_frob > 0.0 { vk_sv_max / gram_x_frob } else { 0.0 };
            eprintln!("  [VK_PROFILE] gram_vk: sv_max/||G_x||_F = {frob_ratio:.3e} (noise guard: complement_is_noise={complement_is_noise})");
            eprintln!("  [VK_PROFILE] gram_vk: retained_rank={retained_rank_vk}, threshold={vk_threshold:.3e}");
            eprintln!("  [VK_PROFILE] gram_vk: leading 5 eigenvalues = {leading:?}");
        }

        let n_basis = retained_rank_vk.min(max_rank);
        let u_vk = eig_vk.u();

        // basis_matrix shape: (rank, n_assess) -- basis vectors are ROWS.
        // Eigenvector k from faer is column k of u_vk; we transpose into
        // row k of the output so that B * B^T = I_rank (orthonormality).
        let mut basis = DMatrix::zeros(n_basis, n_assess);
        for (k, &(_, eig_idx)) in sv_pairs.iter().take(n_basis).enumerate() {
            for col in 0..n_assess {
                basis[(k, col)] = u_vk.read(col, eig_idx);
            }
        }

        // Return up to 2*max_rank SVs for diagnostic gap analysis
        let svs: Vec<f64> = sv_pairs.iter()
            .take(retained_rank_vk.min(max_rank * 2))
            .map(|&(sv, _)| sv)
            .collect();

        (basis, svs)
    });

    (basis_matrix, singular_values, assessors)
}

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
