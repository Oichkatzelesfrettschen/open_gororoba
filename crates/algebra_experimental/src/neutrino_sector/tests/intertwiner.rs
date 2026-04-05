use super::super::*;

/// Intertwiner analysis: is TensorElementLift the unique equivariant map
/// from V_6 to Sym_3(R) under the SU(3) stabilizer action?
///
/// Steps:
/// 1. Build stabilizer action on 42-assessor space (extend G2 derivations
///    from O to S via CD doubling: D(a,b) = (D(a), D(b)))
/// 2. Restrict to V_6 to get rho_V6: su(3) -> gl(6)
/// 3. Build action on Sym_3(R) from the 3x3 complex representation
/// 4. Solve intertwining equations L * rho_V6(X) = rho_Sym3(X) * L
/// 5. Compare solution with TensorElementLift
#[test]
fn test_intertwiner_analysis() {
    use gororoba_algebra::lie::{
        g2_stabilizer::{complex_structure, stabilizer_decomposition},
        g2_su3_representation::fundamental_representation,
    };
    use nalgebra::DMatrix;

    let fixed_unit = 1_usize; // fix e_1

    // ===== STEP 1: Build stabilizer action on 42-assessor space =====
    let decomp = stabilizer_decomposition(fixed_unit);
    let n_stab = decomp.stabilizer_basis.len();
    assert_eq!(n_stab, 8, "Stabilizer should be 8-dimensional");

    // The 42 assessors are (low, high) with low in 1..7, high in 9..15,
    // excluding high == low + 8.
    let (v6_basis, _sv, assessors) = extract_v6_basis();
    let n_assess = assessors.len();
    assert_eq!(n_assess, 42);

    // For each stabilizer generator D (8x8 on octonions), extend to 16x16
    // on sedenions via CD doubling: D(a,b) = (D(a), D(b)).
    // Then compute how D transforms each assessor column.
    //
    // An assessor (low, high) corresponds to a direction in the incidence
    // matrix. D acts on the CD products that define the incidence:
    // D(e_b * e_c) = D(e_b) * e_c + e_b * D(e_c) (Leibniz rule).
    // This transforms the incidence row, hence the assessor vector.
    //
    // However, the assessor is defined by which indices are touched by
    // the pairwise products e_b*e_c, e_b*e_d, e_c*e_d of a triad.
    // The derivation action on assessors is indirect: it permutes/rotates
    // the basis elements, which changes which assessor pairs are activated.
    //
    // Simpler approach: For each stabilizer generator, compute its
    // 16x16 action on sedenion basis, then compute how each assessor
    // pair (low, high) transforms.
    //
    // D extended to S: D_16[i][j] = D_8[i][j] for i,j in 0..8
    //                  D_16[i+8][j+8] = D_8[i][j] for i,j in 0..8
    //                  D_16[i][j+8] = 0 and D_16[i+8][j] = 0

    let mut rho_42: Vec<DMatrix<f64>> = vec![DMatrix::<f64>::zeros(n_assess, n_assess); n_stab];

    for (gen_idx, d) in decomp.stabilizer_basis.iter().enumerate() {
        // Build 16x16 sedenion extension of D
        let mut d16 = [[0.0_f64; 16]; 16];
        for i in 0..8 {
            for j in 0..8 {
                d16[i][j] = d.matrix[i][j];
                d16[i + 8][j + 8] = d.matrix[i][j];
            }
        }

        // For each assessor (low, high), compute how D transforms it.
        // The assessor value at position a is: sum of incidence entries
        // that touch index low or high via pairwise products.
        //
        // Linearized action: D transforms e_low -> sum_j d16[low][j] * e_j
        // and e_high -> sum_j d16[high][j] * e_j.
        // The assessor (low, high) picks up contributions from (j, high)
        // and (low, j) assessors weighted by d16[low][j] and d16[high][j].
        //
        // In the linear approximation:
        // D(assessor_{(low,high)}) = sum_j d16[low][j] * assessor_{(j,high)}
        //                          + sum_j d16[high][j] * assessor_{(low,j)}
        //
        // We need to find which assessor index corresponds to (j, high)
        // or (low, j) -- or if no such assessor exists (because the pair
        // doesn't satisfy the assessor constraints).

        let find_assessor = |l: usize, h: usize| -> Option<usize> {
            assessors.iter().position(|&(al, ah)| al == l && ah == h)
        };

        let mut mat = DMatrix::<f64>::zeros(n_assess, n_assess);

        for (a_idx, &(low, high)) in assessors.iter().enumerate() {
            // D transforms e_low: contribution to assessor space
            for j in 1..=7 {
                let coeff = d16[low][j];
                if coeff.abs() < 1e-15 {
                    continue;
                }
                if let Some(target) = find_assessor(j, high) {
                    mat[(target, a_idx)] += coeff;
                }
            }

            // D transforms e_high: contribution to assessor space
            for j in 9..=15 {
                let coeff = d16[high][j];
                if coeff.abs() < 1e-15 {
                    continue;
                }
                if let Some(target) = find_assessor(low, j) {
                    mat[(target, a_idx)] += coeff;
                }
            }
        }

        rho_42[gen_idx] = mat;
    }

    // ===== STEP 2: Restrict to V_6 =====
    // rho_V6[gen] = V_6^T * rho_42[gen] * V_6  (6x6 matrix)
    let n_v6 = v6_basis.nrows(); // 6
    let mut rho_v6: Vec<DMatrix<f64>> = vec![DMatrix::<f64>::zeros(n_v6, n_v6); n_stab];

    for gen_idx in 0..n_stab {
        // V_6 is n_v6 x 42, rho_42 is 42 x 42
        // rho_v6 = V_6 * rho_42 * V_6^T (since V_6 rows are basis vectors)
        let r42 = &rho_42[gen_idx];
        let mut rv6 = DMatrix::<f64>::zeros(n_v6, n_v6);
        for i in 0..n_v6 {
            for j in 0..n_v6 {
                let mut sum = 0.0_f64;
                for a in 0..n_assess {
                    for b in 0..n_assess {
                        sum += v6_basis[(i, a)] * r42[(a, b)] * v6_basis[(j, b)];
                    }
                }
                rv6[(i, j)] = sum;
            }
        }
        rho_v6[gen_idx] = rv6;
    }

    println!("--- INTERTWINER ANALYSIS ---");
    println!("  Stabilizer generators: {}", n_stab);
    println!("  V_6 restricted representations (6x6):");
    for (idx, rv6) in rho_v6.iter().enumerate() {
        let frob: f64 = (0..n_v6)
            .flat_map(|i| (0..n_v6).map(move |j| rv6[(i, j)] * rv6[(i, j)]))
            .sum::<f64>()
            .sqrt();
        println!("    rho_V6[{}]: Frobenius norm = {:.6}", idx, frob);
    }

    // ===== STEP 3: Build action on Sym_3(R) =====
    // Using the 3x3 complex representation from PR2.
    // For generator X with 3x3 complex rep rho_3(X), the action on
    // Sym_3(R) is: rho_Sym(X)(M) = rho_3(X)*M + M*rho_3(X)^T
    //
    // Vectorize Sym_3(R) using basis:
    // {E_11, E_22, E_33, (E_12+E_21), (E_13+E_31), (E_23+E_32)}
    // (unnormalized symmetric basis, 6 elements)

    let _cs = complex_structure(fixed_unit);
    let _rep = fundamental_representation(&decomp, &_cs);

    // Print first rho_V6 matrix
    println!("\n  rho_V6[0] matrix:");
    for i in 0..n_v6 {
        let row: Vec<String> = (0..n_v6)
            .map(|j| format!("{:8.4}", rho_v6[0][(i, j)]))
            .collect();
        println!("    [{}]", row.join(", "));
    }

    // ===== STEP 4: Analyze the V_6 representation =====
    let mut total_frob = 0.0_f64;
    for gen_idx in 0..n_stab {
        let mut frob = 0.0_f64;
        for i in 0..n_v6 {
            for j in 0..n_v6 {
                let v = rho_v6[gen_idx][(i, j)];
                frob += v * v;
            }
        }
        total_frob += frob.sqrt();
    }

    println!("\n  === V_6 REPRESENTATION ANALYSIS ===");
    println!("  Total Frobenius norm of all rho_V6: {:.6e}", total_frob);

    if total_frob < 1e-10 {
        println!("  rho_V6 is TRIVIAL (zero action).");
        println!("  SU(3) stabilizer does not act on V_6.");
        println!("  Any L: V_6 -> Sym_3(R) is an intertwiner.");
        println!("  TensorElementLift is NOT uniquely determined by equivariance.");
        println!("  The stabilizer SU(3) is the COLOR group (acts within generations),");
        println!("  not the FLAVOR group (acts between generations).");
    } else {
        println!("  rho_V6 is NONTRIVIAL (Frobenius = {:.6e}).", total_frob);
        println!("  SU(3) stabilizer acts on V_6.");

        // Compute the quadratic Casimir C_2 = sum_a rho(T_a)^2
        // For the fundamental representation of su(3), C_2 = (4/3)*I
        let mut casimir = DMatrix::<f64>::zeros(n_v6, n_v6);
        for gen_idx in 0..n_stab {
            let r = &rho_v6[gen_idx];
            // r^2 = r * r
            for i in 0..n_v6 {
                for j in 0..n_v6 {
                    for k in 0..n_v6 {
                        casimir[(i, j)] += r[(i, k)] * r[(k, j)];
                    }
                }
            }
        }

        println!("\n  Casimir C_2 = sum_a rho_V6(T_a)^2:");
        for i in 0..n_v6 {
            let row: Vec<String> = (0..n_v6)
                .map(|j| format!("{:8.4}", casimir[(i, j)]))
                .collect();
            println!("    [{}]", row.join(", "));
        }

        // Check if Casimir is proportional to identity
        let diag_avg: f64 = (0..n_v6).map(|i| casimir[(i, i)]).sum::<f64>() / n_v6 as f64;
        let mut off_diag_max = 0.0_f64;
        let mut diag_dev = 0.0_f64;
        for i in 0..n_v6 {
            for j in 0..n_v6 {
                if i == j {
                    diag_dev += (casimir[(i, j)] - diag_avg).abs();
                } else {
                    off_diag_max = off_diag_max.max(casimir[(i, j)].abs());
                }
            }
        }

        println!("  Casimir diagonal average: {:.6}", diag_avg);
        println!("  Diagonal deviation: {:.6e}", diag_dev);
        println!("  Max off-diagonal: {:.6e}", off_diag_max);

        if off_diag_max < 1e-10 && diag_dev < 1e-10 {
            println!(
                "  Casimir = {:.4} * I_6  (PROPORTIONAL TO IDENTITY)",
                diag_avg
            );
            println!("  => V_6 carries an IRREDUCIBLE representation of su(3)");

            // For su(3) irreps, C_2 = (p^2 + q^2 + pq + 3p + 3q)/3
            // where (p,q) is the Dynkin label.
            // Fund. (1,0): C_2 = 4/3 = 1.333
            // Adj. (1,1): C_2 = 3
            // 6-dim (2,0): C_2 = 10/3 = 3.333
            // 6-dim (0,2): C_2 = 10/3 = 3.333
            println!("  Known su(3) Casimir values for dim-6 irreps:");
            println!("    (2,0) symmetric square: C_2 = 10/3 = 3.333");
            println!("    (0,2) anti-sym. square: C_2 = 10/3 = 3.333");
            println!("    adjoint (1,1) restricted: C_2 = 3.0");
            println!("  Measured C_2 = {:.4}", diag_avg);
        } else {
            println!("  Casimir is NOT proportional to I => V_6 is REDUCIBLE");

            // Diagonalize the Casimir to find irrep decomposition
            let casimir_sym = (&casimir + casimir.transpose()) * 0.5;
            let eig = casimir_sym.symmetric_eigen();
            let mut eigenvalues: Vec<f64> = eig.eigenvalues.iter().copied().collect();
            eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap());

            println!("\n  Casimir eigenvalues (sorted):");
            for (idx, ev) in eigenvalues.iter().enumerate() {
                println!("    lambda[{}] = {:.6}", idx, ev);
            }

            // Check for trivial singlet (eigenvalue = 0)
            let n_singlet = eigenvalues.iter().filter(|ev| ev.abs() < 0.01).count();
            println!("  Trivial SU(3) singlet dimensions: {}", n_singlet);

            if n_singlet > 0 {
                println!("  => V_6 CONTAINS a trivial SU(3) summand!");
                println!("  An SU(3)-invariant lift into flavor space is possible.");
            } else {
                println!("  => V_6 has NO trivial SU(3) summand.");
                println!("  SU(3)-equivariant lift to flavor-only target is impossible.");
                println!("  The right symmetry for the lift is S_3, not SU(3).");
            }
        }
    }
}

/// Casimir eigenvalue decomposition + S_3 intertwiner analysis.
///
/// (1) Diagonalize the su(3) Casimir on V_6 to find irrep decomposition
/// (2) Build the psi (S_3 generator) action on V_6
/// (3) Build the natural S_3 action on Sym_3(R) (generation permutation)
/// (4) Solve the intertwining equation L * rho_V6(psi) = rho_Sym3(psi) * L
/// (5) Compare solution with TensorElementLift
#[test]
fn test_s3_intertwiner_analysis() {
    use cd_kernel::gourlay_psi;
    use nalgebra::DMatrix;

    let (v6_basis, _sv, assessors) = extract_v6_basis();
    let n_v6 = v6_basis.nrows(); // 6
    let n_assess = assessors.len(); // 42

    // ===== PART 1: Casimir eigenvalue decomposition =====
    // Recompute rho_V6 for the stabilizer (same as test_intertwiner_analysis)
    // but here we just need the Casimir matrix.
    // Instead of recomputing everything, we can use the Casimir from the
    // previous test. But for self-containment, let's compute the psi action
    // on V_6 directly -- that's what we really need for S_3 intertwining.

    // ===== PART 2: Build psi action on 42-assessor space =====
    //
    // CORRECTED approach: psi is a sedenion automorphism, so
    // psi(e_b * e_c) = psi(e_b) * psi(e_c). We compute the psi
    // action on assessor space by transforming the INCIDENCE MATRIX.
    //
    // For each Type X triad (b,c,d), compute:
    //   1. Original incidence row (which assessors are touched)
    //   2. Psi-transformed incidence row (apply psi to basis elements,
    //      recompute CD products, find which assessors are touched)
    //   3. The 42x42 transformation is: (X_psi^T * X_original) * pinv(X_original^T * X_original)
    //      where X_original and X_psi have the same rows in corresponding order.
    //
    // Simpler: since both X_original and X_psi have the same column space
    // structure (42 assessors), we can directly compute the column
    // transformation by comparing how each assessor column changes when
    // all triads are psi-transformed.

    // Build the psi-transformed incidence for each assessor:
    // For each assessor (low, high), count how many Type X triad products
    // touch index `low` or `high` BEFORE and AFTER psi transformation.
    //
    // The assessor column vector is: for each triad row, does the triad's
    // CD products touch this assessor's indices?
    //
    // Under psi: triad (b,c,d) -> (b',c',d') where psi(e_b) is a linear
    // combination. For unit basis elements in sedenions, psi maps
    // e_k -> a specific 16D vector.

    // Build 16x16 psi matrix
    let mut psi_mat = [[0.0_f64; 16]; 16];
    for k in 0..16 {
        let mut ek = [0.0_f64; 16];
        ek[k] = 1.0;
        let psi_ek = gourlay_psi(&ek);
        for j in 0..16 {
            psi_mat[j][k] = psi_ek[j];
        }
    }

    // For each assessor pair, compute how psi transforms the indicator.
    // An assessor (low, high) is activated when a CD product output
    // has index = low or index = high.
    //
    // Under psi, basis index m maps to psi(e_m) = sum_j P[j][m] * e_j.
    // So if a CD product outputs e_m, the psi-transformed product outputs
    // sum_j P[j][m] * e_j, which activates assessors containing index j
    // with weight P[j][m].
    //
    // The 42x42 psi action: assessor(low, high) gets weight from
    // assessor(low', high') via the SINGLE-INDEX psi transformation:
    //   T[dst, src] += P[dst_low][src_low]  (if dst_high == src_high)
    //                + P[dst_high][src_high] (if dst_low == src_low)
    //
    // Wait -- that's still wrong. The assessor tests for low OR high
    // independently. The correct single-index action:
    //
    // Under psi, "test for index m" becomes "test for index j with weight P[j][m]".
    // Assessor (low, high) = "test for low" + "test for high" (union).
    // The psi-transformed assessor tests for:
    //   sum_j P[j][low] * (test for j) + sum_j P[j][high] * (test for j)
    // = sum_j (P[j][low] + P[j][high]) * (test for j)
    //
    // Each "test for j" activates ALL assessors whose pair contains j.
    //
    // So T[dst, src] = sum over j in {dst_low, dst_high} of
    //                  (P[j][src_low] + P[j][src_high])
    //
    // But this DOUBLE COUNTS when j appears in both src_low and src_high
    // positions. For distinct indices (which is always the case since
    // low < high and they're in different ranges), this is fine.

    let mut psi_42 = DMatrix::<f64>::zeros(n_assess, n_assess);

    for (src, &(src_low, src_high)) in assessors.iter().enumerate() {
        for (dst, &(dst_low, dst_high)) in assessors.iter().enumerate() {
            // Weight = how much psi maps src's indicator into dst's indicator
            let w = psi_mat[dst_low][src_low]
                + psi_mat[dst_low][src_high]
                + psi_mat[dst_high][src_low]
                + psi_mat[dst_high][src_high];
            if w.abs() > 1e-15 {
                psi_42[(dst, src)] += w;
            }
        }
    }

    // Restrict psi to V_6: rho_V6(psi) = V_6 * psi_42 * V_6^T
    let mut rho_v6_psi = DMatrix::<f64>::zeros(n_v6, n_v6);
    for i in 0..n_v6 {
        for j in 0..n_v6 {
            let mut sum = 0.0_f64;
            for a in 0..n_assess {
                for b in 0..n_assess {
                    sum += v6_basis[(i, a)] * psi_42[(a, b)] * v6_basis[(j, b)];
                }
            }
            rho_v6_psi[(i, j)] = sum;
        }
    }

    println!("--- S_3 INTERTWINER ANALYSIS ---");
    println!("\n  rho_V6(psi) matrix (6x6):");
    for i in 0..n_v6 {
        let row: Vec<String> = (0..n_v6)
            .map(|j| format!("{:8.4}", rho_v6_psi[(i, j)]))
            .collect();
        println!("    [{}]", row.join(", "));
    }

    // Check if rho_V6(psi) is nontrivial
    let mut psi_frob = 0.0_f64;
    for i in 0..n_v6 {
        for j in 0..n_v6 {
            psi_frob += rho_v6_psi[(i, j)] * rho_v6_psi[(i, j)];
        }
    }
    psi_frob = psi_frob.sqrt();
    println!("  |rho_V6(psi)| = {:.6}", psi_frob);

    // Check psi^3 = I on V_6
    let psi2 = &rho_v6_psi * &rho_v6_psi;
    let psi3 = &psi2 * &rho_v6_psi;
    let identity = DMatrix::<f64>::identity(n_v6, n_v6);
    let psi3_error: f64 = (&psi3 - &identity)
        .iter()
        .map(|x| x * x)
        .sum::<f64>()
        .sqrt();
    println!(
        "  |psi^3 - I| = {:.6e} (should be ~0 for order-3)",
        psi3_error
    );

    // ===== PART 3: Build S_3 action on Sym_3(R) =====
    // The natural S_3 action on symmetric 3x3 matrices is by simultaneous
    // row and column permutation. For the order-3 generator psi:
    // psi acts as the cyclic permutation (1 2 3) on generations.
    //
    // On Sym_3(R) vectorized as {M_11, M_22, M_33, M_12, M_13, M_23}:
    // psi(1->2, 2->3, 3->1) maps:
    //   M_11 -> M_22, M_22 -> M_33, M_33 -> M_11  (diagonal cycle)
    //   M_12 -> M_23, M_13 -> M_12, M_23 -> M_13  (off-diagonal cycle... wait)
    //
    // Actually: if psi sends gen i -> gen (i mod 3) + 1, then
    //   M_{ij} -> M_{psi(i), psi(j)}
    // With psi = (1->2, 2->3, 3->1):
    //   M_11 -> M_22, M_22 -> M_33, M_33 -> M_11
    //   M_12 -> M_23, M_23 -> M_31 = M_13, M_13 -> M_21 = M_12
    //
    // So in the basis {M_11, M_22, M_33, M_12, M_13, M_23}:
    let mut rho_sym3_psi = DMatrix::<f64>::zeros(6, 6);
    // Diagonal block: (M_11 -> M_22, M_22 -> M_33, M_33 -> M_11)
    rho_sym3_psi[(1, 0)] = 1.0; // M_11 -> M_22
    rho_sym3_psi[(2, 1)] = 1.0; // M_22 -> M_33
    rho_sym3_psi[(0, 2)] = 1.0; // M_33 -> M_11
    // Off-diagonal block: (M_12 -> M_23, M_13 -> M_12, M_23 -> M_13)
    // In our basis: index 3 = M_12, index 4 = M_13, index 5 = M_23
    rho_sym3_psi[(5, 3)] = 1.0; // M_12 -> M_23
    rho_sym3_psi[(3, 4)] = 1.0; // M_13 -> M_12
    rho_sym3_psi[(4, 5)] = 1.0; // M_23 -> M_13

    println!("\n  rho_Sym3(psi) matrix (6x6, generation permutation):");
    for i in 0..6 {
        let row: Vec<String> = (0..6)
            .map(|j| format!("{:5.1}", rho_sym3_psi[(i, j)]))
            .collect();
        println!("    [{}]", row.join(", "));
    }

    // Verify psi^3 = I on Sym_3
    let sym_psi2 = &rho_sym3_psi * &rho_sym3_psi;
    let sym_psi3 = &sym_psi2 * &rho_sym3_psi;
    let sym_id = DMatrix::<f64>::identity(6, 6);
    let sym_psi3_error: f64 = (&sym_psi3 - &sym_id)
        .iter()
        .map(|x| x * x)
        .sum::<f64>()
        .sqrt();
    println!("  |psi^3 - I| on Sym_3 = {:.6e}", sym_psi3_error);

    // ===== PART 4: Solve intertwining equation =====
    // L * rho_V6(psi) = rho_Sym3(psi) * L
    // where L is 6x6 (maps V_6 -> Sym_3(R)).
    //
    // Vectorize: let l = vec(L) be the 36-element column vector.
    // The equation becomes: (rho_V6(psi)^T kron I_6 - I_6 kron rho_Sym3(psi)) * l = 0
    //
    // For psi alone (one equation): 36 unknowns, 36 equations.
    // Also add psi^2 for redundancy (same equation with psi^2).

    let n = 6_usize;
    let n_sq = n * n; // 36

    // Build the constraint matrix A where A * vec(L) = 0
    // From L * R_V6 = R_S3 * L, we get:
    // (R_V6^T kron I) - (I kron R_S3) applied to vec(L) = 0
    //
    // Kronecker product: (A kron B)_{(i*n+k), (j*n+l)} = A_{ij} * B_{kl}
    // vec(L) maps L_{ij} -> index i*n + j

    let mut constraint = DMatrix::<f64>::zeros(n_sq, n_sq);

    // Add constraint from psi
    for i in 0..n {
        for j in 0..n {
            let row = i * n + j;
            // L * R_V6: sum_k L_{ik} * R_V6_{kj}
            // In vec form: coefficient of L_{ik} at row (i,j) is R_V6_{kj}
            // -> (R_V6^T)_{jk} at position (i*n+j, i*n+k)... wait, let me be
            // more careful.
            //
            // [L * R_V6]_{ij} = sum_k L_{ik} R_V6_{kj}
            // [R_S3 * L]_{ij} = sum_k R_S3_{ik} L_{kj}
            //
            // Setting them equal: sum_k L_{ik} R_V6_{kj} - sum_k R_S3_{ik} L_{kj} = 0
            //
            // In terms of vec(L) where L_{ab} = l[a*n + b]:
            // sum_k l[i*n + k] * R_V6_{kj} - sum_k R_S3_{ik} * l[k*n + j] = 0

            for k in 0..n {
                // From L * R_V6 term:
                let col_1 = i * n + k;
                constraint[(row, col_1)] += rho_v6_psi[(k, j)];

                // From -R_S3 * L term:
                let col_2 = k * n + j;
                constraint[(row, col_2)] -= rho_sym3_psi[(i, k)];
            }
        }
    }

    // Also add constraint from psi^2
    let mut constraint2 = DMatrix::<f64>::zeros(n_sq, n_sq);
    for i in 0..n {
        for j in 0..n {
            let row = i * n + j;
            for k in 0..n {
                let col_1 = i * n + k;
                constraint2[(row, col_1)] += psi2[(k, j)];
                let col_2 = k * n + j;
                constraint2[(row, col_2)] -= sym_psi2[(i, k)];
            }
        }
    }

    // Stack both constraint matrices (72 equations, 36 unknowns)
    let mut full_constraint = DMatrix::<f64>::zeros(2 * n_sq, n_sq);
    for i in 0..n_sq {
        for j in 0..n_sq {
            full_constraint[(i, j)] = constraint[(i, j)];
            full_constraint[(n_sq + i, j)] = constraint2[(i, j)];
        }
    }

    // SVD to find null space
    let constraint_rows = full_constraint.nrows();
    let constraint_cols = full_constraint.ncols();
    let svd = full_constraint.svd(false, true);
    let sigma = &svd.singular_values;
    let v_t = svd.v_t.as_ref().unwrap();

    // Count near-zero singular values (null space dimension)
    let sv_threshold = 1e-8 * sigma[0];
    let null_dim = sigma.iter().filter(|&&s| s < sv_threshold).count();

    println!("\n  === S_3 INTERTWINING EQUATION ===");
    println!(
        "  Constraint matrix: {}x{}",
        constraint_rows, constraint_cols
    );
    println!("  Singular values (last 10):");
    let n_sv = sigma.len();
    for i in (n_sv.saturating_sub(10))..n_sv {
        println!("    sigma[{}] = {:.6e}", i, sigma[i]);
    }
    println!(
        "  Null space dimension: {} (1 = unique equivariant map up to scale)",
        null_dim
    );

    if null_dim > 0 {
        // Extract the null-space vectors (intertwiners)
        // The last null_dim rows of V^T are the null vectors
        println!("\n  Intertwiner(s) found!");

        for ns_idx in 0..null_dim {
            let row_idx = n_sv - 1 - ns_idx;
            if sigma[row_idx] > sv_threshold {
                break;
            }

            // Extract L from vec(L)
            let mut l_mat = DMatrix::<f64>::zeros(n, n);
            for i in 0..n {
                for j in 0..n {
                    l_mat[(i, j)] = v_t[(row_idx, i * n + j)];
                }
            }

            println!("\n  Intertwiner L_{} (6x6, maps V_6 -> Sym_3):", ns_idx);
            for i in 0..n {
                let row: Vec<String> = (0..n).map(|j| format!("{:8.4}", l_mat[(i, j)])).collect();
                println!("    [{}]", row.join(", "));
            }

            // Compare with TensorElementLift's effective matrix
            // TensorElementLift sums assessors in blocks of 7:
            // Block k maps to Sym_3 element k.
            // The effective L_TEL is: L_TEL[sym_idx, v6_idx] = sum over assessors
            // in block sym_idx of v6_basis[v6_idx, assessor]
            let mut l_tel = DMatrix::<f64>::zeros(n, n);
            for sym_idx in 0..6 {
                let block_start = sym_idx * 7;
                let block_end = (block_start + 7).min(42);
                for v6_idx in 0..n_v6 {
                    let mut sum = 0.0_f64;
                    for a in block_start..block_end {
                        sum += v6_basis[(v6_idx, a)];
                    }
                    l_tel[(sym_idx, v6_idx)] = sum;
                }
            }

            // Normalize both for comparison
            let norm_l: f64 = l_mat.iter().map(|x| x * x).sum::<f64>().sqrt();
            let norm_tel: f64 = l_tel.iter().map(|x| x * x).sum::<f64>().sqrt();

            if norm_l > 1e-10 && norm_tel > 1e-10 {
                let l_normed = &l_mat * (1.0 / norm_l);
                let l_tel_normed = &l_tel * (1.0 / norm_tel);

                // Cosine similarity
                let dot: f64 = l_normed
                    .iter()
                    .zip(l_tel_normed.iter())
                    .map(|(a, b)| a * b)
                    .sum();

                println!("\n  Comparison with TensorElementLift:");
                println!("    cos(L_intertwiner, L_TEL) = {:.6}", dot);
                if dot.abs() > 0.95 {
                    println!(
                        "    MATCH: TensorElementLift IS the S_3-equivariant map (up to scale)!"
                    );
                } else if dot.abs() > 0.5 {
                    println!("    PARTIAL: significant overlap but not identical");
                } else {
                    println!("    MISMATCH: TensorElementLift is NOT the equivariant map");
                }
            }
        }
    } else {
        println!(
            "  No intertwiner exists: V_6 and Sym_3(R) carry incompatible S_3 representations."
        );
    }
}

/// 2D constrained scan: optimize theta_12 AND theta_23 simultaneously.
///
/// Finds two orthogonal constrained directions in V_6:
///   u_solar: max g_12.u subject to g_13.u = 0, g_23.u = 0
///   u_atmo:  max g_23.u subject to g_13.u = 0, u_solar.u = 0
/// Then scans over (t1, t2) to push both angles toward PDG.
///
/// Runtime: ~56s. Marked #[ignore] for CI.
/// Run: cargo test -- test_v6_2d_constrained --ignored --nocapture
#[test]
#[ignore]
fn test_v6_2d_constrained_scan() {
    let ch_pair = (11_usize, 12);
    let nu_pair = (7_usize, 8);
    let alpha_ch = 3.75;
    let alpha_nu = 1.30;
    let pdg_t12 = 33.41_f64;
    let pdg_t13 = 8.54_f64;
    let pdg_t23 = 49.0_f64;
    let eps = 0.05_f64;

    let (v6_basis, _sv, _assessors) = extract_v6_basis();
    let lift = TensorElementLift;
    let n_basis = v6_basis.nrows().min(6);

    // Lock baseline permutation
    let (m_ch_0, m_nu_0) = construct_pmns_matrices_two_param(ch_pair, nu_pair, alpha_ch, alpha_nu);
    let eig_ch_0 = m_ch_0.self_adjoint_eigen(faer::Side::Lower).unwrap();
    let eig_nu_0 = m_nu_0.self_adjoint_eigen(faer::Side::Lower).unwrap();
    let u_raw_0 = eig_ch_0.U().transpose() * eig_nu_0.U();
    let (_, perm_u, perm_d) = crate::quark_sector::extract_ckm_permutation_aware(&u_raw_0);

    let compute_angles = |beta: &[f64; 6]| -> (f64, f64, f64) {
        let (m_ch, mut m_nu) =
            construct_pmns_matrices_two_param(ch_pair, nu_pair, alpha_ch, alpha_nu);
        apply_v6_perturbation(&mut m_nu, &v6_basis, beta, &lift);
        let eig_ch = m_ch.self_adjoint_eigen(faer::Side::Lower).unwrap();
        let m_nu_s = (&m_nu + m_nu.transpose()) * faer::Scale(0.5);
        let eig_nu = m_nu_s.self_adjoint_eigen(faer::Side::Lower).unwrap();
        let u_raw = eig_ch.U().transpose() * eig_nu.U();
        let mut u_pmns = faer::Mat::<f64>::zeros(3, 3);
        for i in 0..3 {
            for j in 0..3 {
                u_pmns[(i, j)] = u_raw[(perm_u[i], perm_d[j])];
            }
        }
        extract_pmns_angles(&u_pmns)
    };

    // Compute gradients at beta=0
    let mut g_12 = [0.0_f64; 6];
    let mut g_13 = [0.0_f64; 6];
    let mut g_23 = [0.0_f64; 6];
    for mu in 0..n_basis {
        let mut bp = [0.0_f64; 6];
        bp[mu] = eps;
        let mut bm = [0.0_f64; 6];
        bm[mu] = -eps;
        let (t12_p, t13_p, t23_p) = compute_angles(&bp);
        let (t12_m, t13_m, t23_m) = compute_angles(&bm);
        g_12[mu] = (t12_p - t12_m) / (2.0 * eps);
        g_13[mu] = (t13_p - t13_m) / (2.0 * eps);
        g_23[mu] = (t23_p - t23_m) / (2.0 * eps);
    }

    let dot =
        |a: &[f64; 6], b: &[f64; 6]| -> f64 { a.iter().zip(b.iter()).map(|(x, y)| x * y).sum() };

    // Solar direction: max g_12 subject to g_13 = 0, g_23 = 0
    let u_solar = compute_constrained_solar_direction(&g_12, &g_13, &g_23);
    // Atmospheric direction: max g_23 subject to g_13 = 0, u_solar = 0
    let u_atmo = compute_constrained_atmospheric_direction(&g_23, &g_13, &u_solar);

    let g12_solar = dot(&g_12, &u_solar);
    let g13_solar = dot(&g_13, &u_solar);
    let g23_solar = dot(&g_23, &u_solar);

    let g12_atmo = dot(&g_12, &u_atmo);
    let g13_atmo = dot(&g_13, &u_atmo);
    let g23_atmo = dot(&g_23, &u_atmo);

    println!("--- V_6 2D CONSTRAINED SCAN ---");
    println!(
        "  u_solar = [{}]",
        u_solar
            .iter()
            .map(|x| format!("{:.4}", x))
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!(
        "  u_atmo  = [{}]",
        u_atmo
            .iter()
            .map(|x| format!("{:.4}", x))
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!("\n  Solar direction sensitivity:");
    println!("    g_12.u = {:.6} (solar)", g12_solar);
    println!("    g_13.u = {:.6e} (reactor)", g13_solar);
    println!("    g_23.u = {:.6e} (atmospheric)", g23_solar);
    println!("  Atmospheric direction sensitivity:");
    println!("    g_12.u = {:.6} (solar cross-talk)", g12_atmo);
    println!("    g_13.u = {:.6e} (reactor)", g13_atmo);
    println!("    g_23.u = {:.6} (atmospheric)", g23_atmo);
    println!(
        "  u_solar . u_atmo = {:.6e} (orthogonality)",
        dot(&u_solar, &u_atmo)
    );

    // 2D scan: beta = t1 * u_solar + t2 * u_atmo
    println!("\n  2D scan (t1=solar, t2=atmo):");
    println!(
        "  {:>6} {:>6} {:>10} {:>10} {:>10} {:>10}",
        "t1", "t2", "theta_12", "theta_13", "theta_23", "score"
    );

    let mut best_t1 = 0.0_f64;
    let mut best_t2 = 0.0_f64;
    let mut best_score = f64::MAX;
    let mut best_angles = (0.0_f64, 0.0_f64, 0.0_f64);

    // Coarse grid: t1 in [0, 5], t2 in [-5, 5]
    for step1 in 0..=100_i32 {
        let t1 = step1 as f64 * 0.05;
        for step2 in -100..=100_i32 {
            let t2 = step2 as f64 * 0.05;

            let mut beta = [0.0_f64; 6];
            for k in 0..6 {
                beta[k] = t1 * u_solar[k] + t2 * u_atmo[k];
            }

            let (t12, t13, t23) = compute_angles(&beta);

            // Hard constraint: theta_13 within 0.5 deg
            if (t13 - pdg_t13).abs() > 0.5 {
                continue;
            }

            let score = ((t12 - pdg_t12) / pdg_t12).powi(2)
                + ((t23 - pdg_t23) / pdg_t23).powi(2)
                + 5.0 * ((t13 - pdg_t13) / pdg_t13).powi(2);

            if score < best_score {
                best_score = score;
                best_t1 = t1;
                best_t2 = t2;
                best_angles = (t12, t13, t23);
            }
        }
    }

    // Fine grid around the best point
    let t1_center = best_t1;
    let t2_center = best_t2;
    for step1 in -50..=50_i32 {
        let t1 = t1_center + step1 as f64 * 0.01;
        if t1 < 0.0 {
            continue;
        }
        for step2 in -50..=50_i32 {
            let t2 = t2_center + step2 as f64 * 0.01;

            let mut beta = [0.0_f64; 6];
            for k in 0..6 {
                beta[k] = t1 * u_solar[k] + t2 * u_atmo[k];
            }

            let (t12, t13, t23) = compute_angles(&beta);
            if (t13 - pdg_t13).abs() > 0.5 {
                continue;
            }

            let score = ((t12 - pdg_t12) / pdg_t12).powi(2)
                + ((t23 - pdg_t23) / pdg_t23).powi(2)
                + 5.0 * ((t13 - pdg_t13) / pdg_t13).powi(2);

            if score < best_score {
                best_score = score;
                best_t1 = t1;
                best_t2 = t2;
                best_angles = (t12, t13, t23);
            }
        }
    }

    println!("\n  === 2D CONSTRAINED OPTIMUM ===");
    println!("  t1_solar = {:.4}, t2_atmo = {:.4}", best_t1, best_t2);
    println!(
        "  theta_12 = {:.4} deg (PDG: {:.2}, error: {:.2}%)",
        best_angles.0,
        pdg_t12,
        ((best_angles.0 - pdg_t12) / pdg_t12 * 100.0).abs()
    );
    println!(
        "  theta_13 = {:.4} deg (PDG: {:.2}, error: {:.2}%)",
        best_angles.1,
        pdg_t13,
        ((best_angles.1 - pdg_t13) / pdg_t13 * 100.0).abs()
    );
    println!(
        "  theta_23 = {:.4} deg (PDG: {:.2}, error: {:.2}%)",
        best_angles.2,
        pdg_t23,
        ((best_angles.2 - pdg_t23) / pdg_t23 * 100.0).abs()
    );
    println!("  Combined score: {:.6}", best_score);

    // Report the 4-parameter model
    println!("\n  Full 4-parameter model:");
    println!("    alpha_ch = {:.2}", alpha_ch);
    println!("    alpha_nu = {:.2}", alpha_nu);
    println!("    t_solar  = {:.4}", best_t1);
    println!("    t_atmo   = {:.4}", best_t2);
}

/// Joint 4D optimization: (alpha_ch, alpha_nu, t_solar, t_atmo).
///
/// Re-optimizes the psi coupling parameters jointly with V_6 corrections.
/// The constrained directions are recomputed at each (alpha_ch, alpha_nu)
/// for correctness.
///
/// Runtime: ~160s (Rayon-parallel). Marked #[ignore] for CI.
/// Run: cargo test -- test_v6_joint_4d --ignored --nocapture
#[test]
#[ignore]
fn test_v6_joint_4d_optimization() {
    use rayon::prelude::*;

    let pdg_t12 = 33.41_f64;
    let pdg_t13 = 8.54_f64;
    let pdg_t23 = 49.0_f64;
    let ch_pair = (11_usize, 12);
    let nu_pair = (7_usize, 8);
    let eps = 0.05_f64;

    let (v6_basis, _sv, _assessors) = extract_v6_basis();
    let lift = TensorElementLift;
    let n_basis = v6_basis.nrows().min(6);

    // Helper: for given (alpha_ch, alpha_nu), compute constrained directions
    // and scan (t_solar, t_atmo) to find best angles.
    //
    // OPTIMIZED (3 levels):
    // 1. Precompute M_ch eigenvectors + M_nu baseline ONCE per outer point
    // 2. Precompute perturbation matrices A, B from constrained directions
    //    so inner loop is M_nu(t1,t2) = M_nu_base + t1*A + t2*B (no V_6 recompute)
    // 3. Gradient-guided scan center (Newton estimate of t1)
    let evaluate = |alpha_ch: f64, alpha_nu: f64| -> (f64, (f64, f64, f64), f64, f64) {
        let (m_ch_base, m_nu_base) =
            construct_pmns_matrices_two_param(ch_pair, nu_pair, alpha_ch, alpha_nu);
        let eig_ch = m_ch_base.self_adjoint_eigen(faer::Side::Lower).unwrap();
        let u_ch = eig_ch.U();
        let u_raw_0 = {
            let eig_nu_0 = m_nu_base.self_adjoint_eigen(faer::Side::Lower).unwrap();
            u_ch.transpose() * eig_nu_0.U()
        };
        let (_, perm_u, perm_d) = crate::quark_sector::extract_ckm_permutation_aware(&u_raw_0);

        // Angle extraction from perturbed M_nu (reusing M_ch eigenvectors)
        let angles_from_mnu = |m_nu: &faer::Mat<f64>| -> (f64, f64, f64) {
            let m_nu_s = (m_nu + m_nu.transpose()) * faer::Scale(0.5);
            let eig_nu = m_nu_s.self_adjoint_eigen(faer::Side::Lower).unwrap();
            let u_raw = u_ch.transpose() * eig_nu.U();
            let mut u_pmns = faer::Mat::<f64>::zeros(3, 3);
            for i in 0..3 {
                for j in 0..3 {
                    u_pmns[(i, j)] = u_raw[(perm_u[i], perm_d[j])];
                }
            }
            extract_pmns_angles(&u_pmns)
        };

        // Compute gradients using V_6 basis directions (12 evals)
        let mut g_12 = [0.0_f64; 6];
        let mut g_13 = [0.0_f64; 6];
        let mut g_23 = [0.0_f64; 6];
        for mu in 0..n_basis {
            let mut bp = [0.0_f64; 6];
            bp[mu] = eps;
            let mut bm = [0.0_f64; 6];
            bm[mu] = -eps;
            let mut m_nu_p = m_nu_base.clone();
            let mut m_nu_m = m_nu_base.clone();
            apply_v6_perturbation(&mut m_nu_p, &v6_basis, &bp, &lift);
            apply_v6_perturbation(&mut m_nu_m, &v6_basis, &bm, &lift);
            let (t12_p, t13_p, t23_p) = angles_from_mnu(&m_nu_p);
            let (t12_m, t13_m, t23_m) = angles_from_mnu(&m_nu_m);
            g_12[mu] = (t12_p - t12_m) / (2.0 * eps);
            g_13[mu] = (t13_p - t13_m) / (2.0 * eps);
            g_23[mu] = (t23_p - t23_m) / (2.0 * eps);
        }

        let u_solar = compute_constrained_solar_direction(&g_12, &g_13, &g_23);
        let u_atmo = compute_constrained_atmospheric_direction(&g_23, &g_13, &u_solar);

        // Precompute perturbation matrices A (solar) and B (atmospheric)
        // A = TensorElementLift applied to sum_k u_solar[k] * v6_basis.row(k)
        // B = TensorElementLift applied to sum_k u_atmo[k] * v6_basis.row(k)
        let precompute_perturbation = |u: &[f64; 6]| -> faer::Mat<f64> {
            let mut m_perturbed = m_nu_base.clone();
            let mut beta = [0.0_f64; 6];
            for k in 0..6 {
                beta[k] = u[k];
            }
            apply_v6_perturbation(&mut m_perturbed, &v6_basis, &beta, &lift);
            // A = m_perturbed - m_nu_base
            let mut delta = faer::Mat::<f64>::zeros(3, 3);
            for i in 0..3 {
                for j in 0..3 {
                    delta[(i, j)] = m_perturbed[(i, j)] - m_nu_base[(i, j)];
                }
            }
            delta
        };

        let a_mat = precompute_perturbation(&u_solar);
        let b_mat = precompute_perturbation(&u_atmo);

        // Inner optimization via Gauss-Newton (replaces 651-point grid scan)
        // Closure: (t1, t2) -> (theta_12, theta_13, theta_23) using affine M_nu
        let inner_angles = |t1: f64, t2: f64| -> (f64, f64, f64) {
            let mut m_nu = m_nu_base.clone();
            for i in 0..3 {
                for j in 0..3 {
                    m_nu[(i, j)] = m_nu[(i, j)] + t1 * a_mat[(i, j)] + t2 * b_mat[(i, j)];
                }
            }
            angles_from_mnu(&m_nu)
        };

        // Gradient-guided initial guess for t1
        let dot6 = |a: &[f64; 6], b: &[f64; 6]| -> f64 {
            a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
        };
        let g12_u = dot6(&g_12, &u_solar);
        let t1_init = if g12_u.abs() > 0.01 {
            (pdg_t12 - 28.5) / g12_u
        } else {
            2.5
        };
        let t1_init = t1_init.clamp(0.0, 5.0);

        let (best_t1, best_t2, best_angles, best_score) = gauss_newton_2d(
            &inner_angles,
            t1_init,
            0.0, // initial t2 guess
            (pdg_t12, pdg_t13, pdg_t23),
            (1.0, 2.24, 1.0), // sqrt(5) weight on theta_13
            15,               // max iterations
        );

        (best_score, best_angles, best_t1, best_t2)
    };

    println!("--- JOINT 4D OPTIMIZATION ---");

    // Coarse grid over (alpha_ch, alpha_nu)
    // Previous optimum: (3.50, 1.35). Focused neighborhood.
    let grid: Vec<(f64, f64)> = (25..=50_i32)
        .flat_map(|i| (8..=20_i32).map(move |j| (i as f64 * 0.1, j as f64 * 0.1)))
        .collect();

    let results: Vec<(f64, f64, f64, (f64, f64, f64), f64, f64)> = grid
        .par_iter()
        .map(|&(a_ch, a_nu)| {
            let (score, angles, t1, t2) = evaluate(a_ch, a_nu);
            (score, a_ch, a_nu, angles, t1, t2)
        })
        .collect();

    let best = results
        .iter()
        .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
        .unwrap();

    println!("  Coarse grid: {} points evaluated", results.len());
    println!(
        "  Best coarse: alpha_ch={:.2}, alpha_nu={:.2}, t1={:.2}, t2={:.2}",
        best.1, best.2, best.4, best.5
    );
    println!(
        "    theta_12 = {:.4} (error {:.2}%)",
        (best.3).0,
        (((best.3).0 - pdg_t12) / pdg_t12 * 100.0).abs()
    );
    println!(
        "    theta_13 = {:.4} (error {:.2}%)",
        (best.3).1,
        (((best.3).1 - pdg_t13) / pdg_t13 * 100.0).abs()
    );
    println!(
        "    theta_23 = {:.4} (error {:.2}%)",
        (best.3).2,
        (((best.3).2 - pdg_t23) / pdg_t23 * 100.0).abs()
    );
    println!("    Score = {:.6}", best.0);

    // Fine refinement around the coarse best
    let a_ch_center = best.1;
    let a_nu_center = best.2;

    let fine_grid: Vec<(f64, f64)> = (-10..=10_i32)
        .flat_map(|i| {
            (-10..=10_i32)
                .map(move |j| (a_ch_center + i as f64 * 0.05, a_nu_center + j as f64 * 0.05))
        })
        .filter(|&(a, b)| a > 0.0 && b > 0.0)
        .collect();

    let fine_results: Vec<(f64, f64, f64, (f64, f64, f64), f64, f64)> = fine_grid
        .par_iter()
        .map(|&(a_ch, a_nu)| {
            let (score, angles, t1, t2) = evaluate(a_ch, a_nu);
            (score, a_ch, a_nu, angles, t1, t2)
        })
        .collect();

    let fine_best = fine_results
        .iter()
        .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
        .unwrap();

    println!("\n  Fine grid: {} points evaluated", fine_results.len());
    println!("\n  === JOINT 4D OPTIMUM ===");
    println!("  alpha_ch = {:.2}", fine_best.1);
    println!("  alpha_nu = {:.2}", fine_best.2);
    println!("  t_solar  = {:.2}", fine_best.4);
    println!("  t_atmo   = {:.2}", fine_best.5);
    println!(
        "  theta_12 = {:.4} deg (PDG: {:.2}, error: {:.2}%)",
        (fine_best.3).0,
        pdg_t12,
        (((fine_best.3).0 - pdg_t12) / pdg_t12 * 100.0).abs()
    );
    println!(
        "  theta_13 = {:.4} deg (PDG: {:.2}, error: {:.2}%)",
        (fine_best.3).1,
        pdg_t13,
        (((fine_best.3).1 - pdg_t13) / pdg_t13 * 100.0).abs()
    );
    println!(
        "  theta_23 = {:.4} deg (PDG: {:.2}, error: {:.2}%)",
        (fine_best.3).2,
        pdg_t23,
        (((fine_best.3).2 - pdg_t23) / pdg_t23 * 100.0).abs()
    );
    println!("  Combined score: {:.6}", fine_best.0);

    // Compare with previous best
    let prev_score = ((33.37 - pdg_t12) / pdg_t12).powi(2)
        + ((47.40 - pdg_t23) / pdg_t23).powi(2)
        + 5.0 * ((8.52 - pdg_t13) / pdg_t13).powi(2);
    println!(
        "\n  Previous 4-param score (3.75, 1.30, 2.49, 0.11): {:.6}",
        prev_score
    );
    println!("  Improvement: {:.1}x", prev_score / fine_best.0);
}
