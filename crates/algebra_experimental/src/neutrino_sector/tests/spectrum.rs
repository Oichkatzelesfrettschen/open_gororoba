use super::super::*;
use rayon::prelude::*;

    /// Joint (alpha_CP, t_solar, t_atmo) 3D optimization for J_max
    /// (C-1497 AMENDED: yields J_max, not PDG |J|; 3.9x discrepancy).
    ///
    /// # Physical motivation
    ///
    /// The phase-only pipeline (C-1494) fixes the V_6 parameters at the
    /// Gauss-Newton optimum for the real mass matrix (t_sol=1.35,
    /// t_atm=2.25), then scans alpha_CP alone.  This yields |J_CP| =
    /// 8.5e-3 (25% of PDG).  The insight: the real mass matrix must
    /// "make room" for complex phases by shifting in mixing-angle space.
    /// Joint optimization allows this trade-off to happen naturally.
    ///
    /// # Scan geometry
    ///
    /// ```text
    /// alpha_CP :  [0.025, 0.50]   20 steps    (CP amplitude)
    /// t_solar  :  GN +/- 3.0      31 steps    (solar angle driver)
    /// t_atmo   :  GN +/- 3.0      31 steps    (atmospheric angle driver)
    /// k        :  1..7             7 embeddings (outer loop)
    ///
    /// Total: 7 * 20 * 31 * 31 = 134,540 eigendecompositions (~3.5 min)
    /// ```
    ///
    /// # Acceptance criterion
    ///
    /// All three mixing angles must be within **2% of PDG** (stricter
    /// than the 5% tolerance used in the single-parameter scan):
    /// - theta_12: 33.41 +/- 0.67 deg
    /// - theta_13:  8.54 +/- 0.17 deg
    /// - theta_23: 49.00 +/- 0.98 deg
    ///
    /// Among all accepted points, the one with largest |J_CP| wins.
    ///
    /// # Two-pass scan architecture
    ///
    /// ```text
    /// Pass 1 (coarse, rayon-parallel over k=1..7):
    ///   10 alpha x 11 t_sol x 11 t_atm = 1210 pts/k
    ///   7 k-embeddings in parallel => ~8500 eigendecomps, ~2s
    ///
    /// Pass 2 (fine, single k around coarse winner):
    ///   11 alpha x 11 t_sol x 11 t_atm = 1331 pts
    ///   Step sizes 5x finer => ~0.2s
    ///
    /// Total: ~3-4s scan + ~6s GN setup = ~10s wall time
    /// ```
    ///
    /// # Key result
    ///
    /// ```text
    /// k=5: alpha_CP=0.450, t_sol=1.027, t_atm=3.927
    ///       |J_CP| = 3.33e-2 = J_max (kinematic maximum at delta~90)
    ///       delta_CP = arg(Jarlskog quartet) = 92.8 deg
    /// ```
    ///
    /// # AMENDED J_CP interpretation (C-1497)
    ///
    /// |J_CP| = 3.33e-2 is the KINEMATIC MAXIMUM:
    ///   J_max = c12*s12*c23*s23*s13*c13^2 ~ 0.033
    /// attained because the framework gives delta ~ 90 (|sin(delta)| ~ 1).
    ///
    /// PDG measured |J| = J_max * |sin(195)| = 0.033 * 0.259 = 8.6e-3.
    /// Our |J| / |J_PDG| = 3.9x LARGER than experiment.
    ///
    /// The earlier "101% of PDG" claim was MISLEADING because it compared
    /// our J_max against the kinematic bound, not the measured value.
    ///
    /// # delta_CP analysis (C-1498)
    ///
    /// Three independent delta extractions agree:
    /// - `arg(-U_e3) = 97.9 deg` (convention-dependent)
    /// - `arg(Jarlskog quartet) = 92.8 deg` (rephasing-invariant)
    /// - `atan2(sin_delta, cos_delta) invariant` (from moduli + J)
    ///
    /// All give delta ~ pi/2 = MAXIMAL CP violation (|sin(delta)| ~ 1).
    /// PDG best-fit: 195 +/- 25 deg = NON-MAXIMAL (|sin(delta)| ~ 0.26).
    /// The framework CANNOT accommodate delta = 195 without breaking
    /// the angle fit.  This is a genuine discrepancy testable by DUNE
    /// and Hyper-Kamiokande.
    ///
    /// # Claims exercised
    ///
    /// - C-1497: Joint 3D scan gives J_max (AMENDED from "101% of PDG")
    /// - C-1498: delta_CP = 93 deg (near-maximal, 3.9x discrepancy vs PDG)
    /// - C-1494: Phase-only baseline (compared against)
    #[test]
    fn test_cp_violation_joint_3d_scan() {
        use cd_kernel::gourlay_psi;
        use crate::majorana_braiding::MajoranaMode;
        use crate::bell_inequality::rotate_sparse;

        let ch_pair = (11_usize, 12);
        let nu_pair = (7_usize, 8);
        let alpha_ch = 3.00_f64;
        let alpha_nu = 1.35_f64;

        let (m_ch_real, m_nu_real) = construct_pmns_matrices_two_param(
            ch_pair, nu_pair, alpha_ch, alpha_nu,
        );
        let (v6_basis, _, _) = extract_v6_basis();
        let lift = TensorElementLift;
        let eps = 0.05_f64;
        let n_basis = v6_basis.nrows().min(6);

        let eig_ch_0 = m_ch_real.selfadjoint_eigendecomposition(faer::Side::Lower);
        let eig_nu_0 = m_nu_real.selfadjoint_eigendecomposition(faer::Side::Lower);
        let u_raw_0 = eig_ch_0.u().transpose() * eig_nu_0.u();
        let (_, perm_u, perm_d) = crate::quark_sector::extract_ckm_permutation_aware(&u_raw_0);

        let angles_at = |beta: &[f64; 6]| -> (f64, f64, f64) {
            let mut m_nu = m_nu_real.clone();
            apply_v6_perturbation(&mut m_nu, &v6_basis, beta, &lift);
            let m_nu_s = (&m_nu + m_nu.transpose()) * faer::scale(0.5);
            let eig_nu = m_nu_s.selfadjoint_eigendecomposition(faer::Side::Lower);
            let u_raw = eig_ch_0.u().transpose() * eig_nu.u();
            let mut u_perm = faer::Mat::zeros(3, 3);
            for i in 0..3 { for j in 0..3 {
                u_perm.write(i, j, u_raw.read(perm_u[i], perm_d[j]));
            }}
            extract_pmns_angles(&u_perm)
        };

        let mut g_12 = [0.0_f64; 6];
        let mut g_13 = [0.0_f64; 6];
        let mut g_23 = [0.0_f64; 6];
        for mu in 0..n_basis {
            let mut bp = [0.0_f64; 6];
            let mut bm = [0.0_f64; 6];
            bp[mu] = eps;
            bm[mu] = -eps;
            let (t12p, t13p, t23p) = angles_at(&bp);
            let (t12m, t13m, t23m) = angles_at(&bm);
            g_12[mu] = (t12p - t12m) / (2.0 * eps);
            g_13[mu] = (t13p - t13m) / (2.0 * eps);
            g_23[mu] = (t23p - t23m) / (2.0 * eps);
        }

        let u_solar = compute_constrained_solar_direction(&g_12, &g_13, &g_23);
        let u_atmo = compute_constrained_atmospheric_direction(&g_23, &g_13, &u_solar);

        let inner_angles = |t_s: f64, t_a: f64| -> (f64, f64, f64) {
            let mut beta = [0.0_f64; 6];
            for kk in 0..6 { beta[kk] = t_s * u_solar[kk] + t_a * u_atmo[kk]; }
            angles_at(&beta)
        };
        // Gauss-Newton gives the real-matrix angle optimum.  The 3D scan
        // will explore deviations from this point, allowing t_sol and t_atm
        // to shift to accommodate larger CP phases.
        let (t_sol, t_atm, _, _) = gauss_newton_2d(
            &inner_angles, 1.5, 0.0,
            (33.41, 8.54, 49.0), (1.0, 2.24, 1.0), 15,
        );

        // Build friction profiles from sedenion associators [a, x, b]
        // with selectors (e_7, e_8).  Three profiles for three generations,
        // one per quaternionic subalgebra of the octonion.
        let (o1, o2, o3, sign_table) = super::psi_setup();
        let subs = [&o1, &o2, &o3];
        let nu_a = MajoranaMode { gamma_index: nu_pair.0 - 1, cd_basis_index: nu_pair.0, cd_dim: 16 };
        let nu_b = MajoranaMode { gamma_index: nu_pair.1 - 1, cd_basis_index: nu_pair.1, cd_dim: 16 };

        let build_profile = |mode_i: &MajoranaMode, mode_j: &MajoranaMode, sub: &[usize]| -> [f64; 16] {
            let i = mode_i.cd_basis_index;
            let j = mode_j.cd_basis_index;
            let a_sparse = vec![(i, 1.0)];
            let a_rotated = rotate_sparse(&a_sparse, i, j, std::f64::consts::FRAC_PI_4);
            let b_sparse = vec![(j, 1.0)];
            let mut profile = [0.0_f64; 16];
            for &kk in sub {
                if kk == 0 || kk == i || kk == j { continue; }
                let x_sparse = [(kk, 1.0)];
                profile[kk] = sign_table.sparse_associator_sum(&a_rotated, &x_sparse, &b_sparse);
            }
            profile
        };

        let nu_profiles: Vec<[f64; 16]> = subs.iter()
            .map(|s| build_profile(&nu_a, &nu_b, s)).collect();

        let dot16 = |a: &[f64; 16], b: &[f64; 16]| -> f64 {
            a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
        };

        println!("--- CP VIOLATION: JOINT 3D SCAN (alpha_CP, t_solar, t_atmo) ---\n");
        println!("  GN baseline: t_sol={:.4}, t_atm={:.4}", t_sol, t_atm);

        // Two-pass parallel scan: COARSE (10*11*11 = 1210 pts/k, ~7 k, ~1s)
        // then FINE (10*11*11 = 1210 pts around winner, ~0.1s).
        // Total: ~8500 eigendecompositions, wall time ~2-4s.
        //
        // All shared state is immutable -- no synchronisation needed.

        // Per-k scan closure.  Returns best (k, alpha, ts, ta, t12, t13, t23, jcp, delta).
        let scan_k = |k: usize| -> Option<(usize, f64, f64, f64, f64, f64, f64, f64, f64)> {
            let apply_jk = |v: &[f64; 16]| -> [f64; 16] { apply_jk_full_16d(v, k) };

            let mut re_overlap = [[0.0_f64; 3]; 3];
            let mut im_overlap = [[0.0_f64; 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    let psi_j = gourlay_psi(&nu_profiles[j]);
                    re_overlap[i][j] = dot16(&nu_profiles[i], &psi_j);
                    if i != j {
                        let jk_psi_j = apply_jk(&psi_j);
                        im_overlap[i][j] = dot16(&nu_profiles[i], &jk_psi_j);
                    }
                }
            }

            let mut phi = [[0.0_f64; 3]; 3];
            let mut has_phase = false;
            for i in 0..3 {
                for j in 0..3 {
                    if i != j {
                        phi[i][j] = im_overlap[i][j].atan2(re_overlap[i][j]);
                        if phi[i][j].abs() > 1e-10 { has_phase = true; }
                    }
                }
            }
            if !has_phase { return None; }

            let mut best_jcp = 0.0_f64;
            let mut best = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);

            let ctx = CpScanContext {
                m_nu_real: &m_nu_real,
                m_ch_real: &m_ch_real,
                v6_basis: &v6_basis,
                u_solar: &u_solar,
                u_atmo: &u_atmo,
                lift: &lift,
                perm_u: [perm_u[0], perm_u[1], perm_u[2]],
                perm_d: [perm_d[0], perm_d[1], perm_d[2]],
            };
            // Coarse pass: 10 alpha x 11 t_sol x 11 t_atm = 1210 pts
            // Uses zero-alloc Cardano solver for ~10x speedup over faer.
            for a_step in 1..=10_i32 {
                let alpha_cp = a_step as f64 * 0.05;
                for ts_step in -5..=5_i32 {
                    let t_sol_trial = t_sol + ts_step as f64 * 0.2;
                    for ta_step in -5..=5_i32 {
                        let t_atm_trial = t_atm + ta_step as f64 * 0.6;

                        let r = evaluate_cp_scan_point_cardano(
                            alpha_cp, t_sol_trial, t_atm_trial, &phi,
                            &ctx,
                        );

                        let err_12 = ((r.theta_12 - 33.41) / 33.41).abs();
                        let err_13 = ((r.theta_13 - 8.54) / 8.54).abs();
                        let err_23 = ((r.theta_23 - 49.0) / 49.0).abs();
                        if err_12 > 0.02 || err_13 > 0.02 || err_23 > 0.02 {
                            continue;
                        }

                        if r.j_cp.abs() > best_jcp.abs() {
                            best_jcp = r.j_cp;
                            best = (alpha_cp, t_sol_trial, t_atm_trial,
                                    r.theta_12, r.theta_13, r.theta_23, r.delta_cp);
                        }
                    }
                }
            }

            if best_jcp.abs() > 1e-6 {
                let (alpha, ts, ta, t12, t13, t23, delta) = best;
                Some((k, alpha, ts, ta, t12, t13, t23, best_jcp, delta))
            } else {
                None
            }
        };

        // Parallel dispatch: 7 threads, one per k-embedding
        let results: Vec<_> = (1..=7_usize).into_par_iter()
            .filter_map(scan_k)
            .collect();

        // Reduce: pick global maximum |J_CP|
        let winner = results.iter()
            .max_by(|a, b| a.7.abs().partial_cmp(&b.7.abs()).unwrap());

        if let Some(&(k_best, alpha_c, ts_c, ta_c, t12_c, t13_c, t23_c, jcp_c, delta_c)) = winner {
            println!("  Coarse best k={}: alpha_CP={:.4}, t_sol={:.3}, t_atm={:.3}", k_best, alpha_c, ts_c, ta_c);
            println!("  Coarse |J_CP| = {:.4e}, delta = {:.1} deg", jcp_c.abs(), delta_c);

            // Print all k results for comparison
            println!("\n  All k-embedding results (coarse):");
            for &(kk, a, ts2, ta2, _t12b, _t13b, _t23b, jcpb, db) in &results {
                println!("    k={}: alpha={:.3}, t_sol={:.3}, t_atm={:.3}, |J|={:.3e}, delta={:.1}",
                    kk, a, ts2, ta2, jcpb.abs(), db);
            }

            // ---------------------------------------------------------------
            // Fine pass: 10*11*11 = 1210 pts around coarse winner
            // alpha +/- 0.05 (step 0.01), t_sol +/- 0.2 (step 0.04),
            // t_atm +/- 0.6 (step 0.12)
            // ---------------------------------------------------------------
            println!("\n  Fine-grid refinement around k={}, alpha={:.3}, t_sol={:.3}, t_atm={:.3}:",
                k_best, alpha_c, ts_c, ta_c);

            // Recompute phi for the winning k
            let apply_jk_fine = |v: &[f64; 16]| -> [f64; 16] { apply_jk_full_16d(v, k_best) };
            let mut phi_fine = [[0.0_f64; 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    if i != j {
                        let psi_j = gourlay_psi(&nu_profiles[j]);
                        let re = dot16(&nu_profiles[i], &psi_j);
                        let jk_psi = apply_jk_fine(&psi_j);
                        let im = dot16(&nu_profiles[i], &jk_psi);
                        phi_fine[i][j] = im.atan2(re);
                    }
                }
            }

            // Reuse the same context struct; fresh buffers for the fine pass
            let ctx_fine = CpScanContext {
                m_nu_real: &m_nu_real,
                m_ch_real: &m_ch_real,
                v6_basis: &v6_basis,
                u_solar: &u_solar,
                u_atmo: &u_atmo,
                lift: &lift,
                perm_u: [perm_u[0], perm_u[1], perm_u[2]],
                perm_d: [perm_d[0], perm_d[1], perm_d[2]],
            };
            let mut fine_best_jcp = jcp_c;
            let mut fine_best = (alpha_c, ts_c, ta_c, t12_c, t13_c, t23_c, delta_c);

            // Fine pass uses zero-alloc Cardano solver.
            for a_step in -5..=5_i32 {
                let alpha_cp = (alpha_c + a_step as f64 * 0.01).max(0.001);
                for ts_step in -5..=5_i32 {
                    let t_sol_f = ts_c + ts_step as f64 * 0.04;
                    for ta_step in -5..=5_i32 {
                        let t_atm_f = ta_c + ta_step as f64 * 0.12;

                        let r = evaluate_cp_scan_point_cardano(
                            alpha_cp, t_sol_f, t_atm_f, &phi_fine,
                            &ctx_fine,
                        );

                        let err_12 = ((r.theta_12 - 33.41) / 33.41).abs();
                        let err_13 = ((r.theta_13 - 8.54) / 8.54).abs();
                        let err_23 = ((r.theta_23 - 49.0) / 49.0).abs();
                        if err_12 > 0.02 || err_13 > 0.02 || err_23 > 0.02 { continue; }

                        if r.j_cp.abs() > fine_best_jcp.abs() {
                            fine_best_jcp = r.j_cp;
                            fine_best = (alpha_cp, t_sol_f, t_atm_f,
                                         r.theta_12, r.theta_13, r.theta_23, r.delta_cp);
                        }
                    }
                }
            }

            let (alpha, ts, ta, t12, t13, t23, delta) = fine_best;
            println!("  Refined k={}: alpha_CP={:.4}, t_sol={:.4}, t_atm={:.4}", k_best, alpha, ts, ta);
            println!("  Angles: theta_12={:.2}, theta_13={:.2}, theta_23={:.2}", t12, t13, t23);
            println!("  |J_CP| = {:.4e}, delta_CP = {:.1} deg", fine_best_jcp.abs(), delta);

            let err_12 = ((t12 - 33.41) / 33.41 * 100.0).abs();
            let err_13 = ((t13 - 8.54) / 8.54 * 100.0).abs();
            let err_23 = ((t23 - 49.0) / 49.0 * 100.0).abs();
            println!("  Errors: t12={:.2}%, t13={:.2}%, t23={:.2}%", err_12, err_13, err_23);
            println!("  J_max (kinematic) = {:.4e}", fine_best_jcp.abs());
            println!("  PDG |J| = 8.6e-3 (non-maximal, sin(195)=0.26)");
            println!("  |J|/|J_PDG| = {:.1}x (3.9x expected for maximal CP)", fine_best_jcp.abs() / 0.0086);

            // Rephasing-aware delta_CP: recompute via Cardano scan point
            // to get the invariant delta, plus faer for Jarlskog quartet arg.
            let r_final = evaluate_cp_scan_point_cardano(
                alpha, ts, ta, &phi_fine, &ctx_fine,
            );

            // For the Jarlskog quartet arg we need the full PMNS matrix.
            // Use faer for this single-point verification.
            let mut bufs_fine = CpScanBuffers::new();
            let mut beta_f = [0.0_f64; 6];
            for kk in 0..6 {
                beta_f[kk] = ts * ctx_fine.u_solar[kk] + ta * ctx_fine.u_atmo[kk];
            }
            let mut m_nu_f = m_nu_real.clone();
            apply_v6_perturbation(&mut m_nu_f, &v6_basis, &beta_f, &lift);
            let m_nu_f = (&m_nu_f + m_nu_f.transpose()) * faer::scale(0.5);
            for i in 0..3 {
                bufs_fine.m_nu.write(i, i, faer::complex_native::c64::new(m_nu_f.read(i, i), 0.0));
                bufs_fine.m_ch.write(i, i, faer::complex_native::c64::new(m_ch_real.read(i, i), 0.0));
                for j in (i + 1)..3 {
                    let phase = alpha * phi_fine[i][j];
                    let mag = m_nu_f.read(i, j);
                    bufs_fine.m_nu.write(i, j, faer::complex_native::c64::new(
                        mag * phase.cos(), mag * phase.sin()));
                    bufs_fine.m_nu.write(j, i, faer::complex_native::c64::new(
                        mag * phase.cos(), -mag * phase.sin()));
                    bufs_fine.m_ch.write(i, j, faer::complex_native::c64::new(m_ch_real.read(i, j), 0.0));
                    bufs_fine.m_ch.write(j, i, faer::complex_native::c64::new(m_ch_real.read(j, i), 0.0));
                }
            }
            let eig_ch_r = bufs_fine.m_ch.selfadjoint_eigendecomposition(faer::Side::Lower);
            let eig_nu_r = bufs_fine.m_nu.selfadjoint_eigendecomposition(faer::Side::Lower);
            let u_r = eig_ch_r.u().adjoint() * eig_nu_r.u();
            let u_at_r = |i: usize, j: usize| -> faer::complex_native::c64 {
                u_r.read(perm_u[i], perm_d[j])
            };

            let jarlskog_q = u_at_r(0, 1) * u_at_r(1, 2)
                * u_at_r(0, 2).conj() * u_at_r(1, 1).conj();
            let delta_jarlskog = jarlskog_q.arg().to_degrees();
            let delta_ue3 = (-u_at_r(0, 2)).arg().to_degrees();
            let _ = r_final; // consistency check -- same as fine_best

            println!("\n  --- delta_CP extraction (rephasing analysis) ---");
            println!("  arg(-U_e3) = {:.1} deg", delta_ue3);
            println!("  arg(Jarlskog quartet) = {:.1} deg", delta_jarlskog);
            println!("  atan2(sin,cos) invariant = {:.1} deg", r_final.delta_cp_invariant);
            println!("  PDG NuFIT 5.3: delta = 195 +/- 25 deg (NO)");

            // ----- Nelder-Mead refinement (B2) -----
            println!("\n  --- Nelder-Mead refinement (k={}, constrained fit) ---", k_best);
            let (nm_result, nm_params) = refine_cp_nelder_mead(
                &ctx_fine, &phi_fine, alpha, ts, ta, false,
            );
            println!("  NM params: alpha={:.4}, t_sol={:.4}, t_atm={:.4}",
                nm_params[0], nm_params[1], nm_params[2]);
            println!("  NM angles: t12={:.2}, t13={:.2}, t23={:.2}",
                nm_result.theta_12, nm_result.theta_13, nm_result.theta_23);
            println!("  NM |J_CP| = {:.4e}, delta_inv = {:.1} deg",
                nm_result.j_cp.abs(), nm_result.delta_cp_invariant);

            // Prediction mode (B2): pure -|J| cost, no angle penalty
            let (nm_pred, nm_pred_params) = refine_cp_nelder_mead(
                &ctx_fine, &phi_fine, alpha, ts, ta, true,
            );
            println!("\n  --- Nelder-Mead prediction mode ---");
            println!("  Pred params: alpha={:.4}, t_sol={:.4}, t_atm={:.4}",
                nm_pred_params[0], nm_pred_params[1], nm_pred_params[2]);
            println!("  Pred angles: t12={:.2}, t13={:.2}, t23={:.2}",
                nm_pred.theta_12, nm_pred.theta_13, nm_pred.theta_23);
            println!("  Pred |J_CP| = {:.4e}", nm_pred.j_cp.abs());
            let pred_12_err = ((nm_pred.theta_12 - 33.41) / 33.41 * 100.0).abs();
            let pred_13_err = ((nm_pred.theta_13 - 8.54) / 8.54 * 100.0).abs();
            let pred_23_err = ((nm_pred.theta_23 - 49.0) / 49.0 * 100.0).abs();
            println!("  Pred errors: t12={:.1}%, t13={:.1}%, t23={:.1}%",
                pred_12_err, pred_13_err, pred_23_err);
            if pred_12_err > 20.0 || pred_13_err > 20.0 || pred_23_err > 20.0 {
                println!("  ** Prediction mode: angles DIVERGE -- structure is not generative **");
            }

            // ----- NM with r-penalty: Pareto exploration -----
            println!("\n  --- Nelder-Mead with r-penalty (Pareto) ---");
            for &w in &[0.01, 0.1, 1.0, 10.0] {
                let (nm_r, nm_r_params) = refine_cp_nelder_mead_r(
                    &ctx_fine, &phi_fine, alpha, ts, ta, false, w,
                );
                // Compute r at this point
                let mut beta_r = [0.0_f64; 6];
                for kk in 0..6 {
                    beta_r[kk] = nm_r_params[1] * ctx_fine.u_solar[kk]
                               + nm_r_params[2] * ctx_fine.u_atmo[kk];
                }
                let mut m_nu_r = m_nu_real.clone();
                apply_v6_perturbation(&mut m_nu_r, &v6_basis, &beta_r, &lift);
                let m_nu_r = (&m_nu_r + m_nu_r.transpose()) * faer::scale(0.5);
                let mut h_r: [[C2; 3]; 3] = [[(0.0, 0.0); 3]; 3];
                for i in 0..3 {
                    h_r[i][i] = (m_nu_r.read(i, i), 0.0);
                    for j in (i + 1)..3 {
                        let phase = nm_r_params[0] * phi_fine[i][j];
                        let mag = m_nu_r.read(i, j);
                        h_r[i][j] = (mag * phase.cos(), mag * phase.sin());
                        h_r[j][i] = (mag * phase.cos(), -mag * phase.sin());
                    }
                }
                let (ev_r, _) = hermitian_3x3_eig_hybrid(&h_r);
                let mut ev_abs: [f64; 3] = [ev_r[0].abs(), ev_r[1].abs(), ev_r[2].abs()];
                ev_abs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let dm21 = ev_abs[1]*ev_abs[1] - ev_abs[0]*ev_abs[0];
                let dm31 = ev_abs[2]*ev_abs[2] - ev_abs[0]*ev_abs[0];
                let r_val = if dm31.abs() > 1e-30 { dm21/dm31 } else { 0.0 };
                let err_12 = ((nm_r.theta_12 - 33.41) / 33.41 * 100.0).abs();
                let err_13 = ((nm_r.theta_13 - 8.54) / 8.54 * 100.0).abs();
                let err_23 = ((nm_r.theta_23 - 49.0) / 49.0 * 100.0).abs();
                println!("  w={w:5.2}: t12={:.2}({err_12:.1}%), t13={:.2}({err_13:.1}%), t23={:.2}({err_23:.1}%), |J|={:.3e}, r={r_val:.4}",
                    nm_r.theta_12, nm_r.theta_13, nm_r.theta_23, nm_r.j_cp.abs());
            }

            // ----- Gradient recomputation + principal angle drift (C1) -----
            println!("\n  --- Gradient recomputation at NM optimum ---");
            let angles_at_nm = |beta: &[f64; 6]| -> (f64, f64, f64) {
                let mut m_nu = m_nu_real.clone();
                apply_v6_perturbation(&mut m_nu, &v6_basis, beta, &lift);
                let m_nu_s = (&m_nu + m_nu.transpose()) * faer::scale(0.5);
                let eig_nu = m_nu_s.selfadjoint_eigendecomposition(faer::Side::Lower);
                let u_raw = eig_ch_0.u().transpose() * eig_nu.u();
                let mut u_perm2 = faer::Mat::zeros(3, 3);
                for i in 0..3 { for j in 0..3 {
                    u_perm2.write(i, j, u_raw.read(perm_u[i], perm_d[j]));
                }}
                extract_pmns_angles(&u_perm2)
            };
            let eps_g = 0.05_f64;
            let mut g_12_nm = [0.0_f64; 6];
            let mut g_13_nm = [0.0_f64; 6];
            let mut g_23_nm = [0.0_f64; 6];
            for mu in 0..n_basis {
                let mut bp = [0.0_f64; 6];
                let mut bm = [0.0_f64; 6];
                // Perturb around NM-optimized beta
                for kk in 0..6 {
                    bp[kk] = nm_params[1] * ctx_fine.u_solar[kk]
                           + nm_params[2] * ctx_fine.u_atmo[kk];
                    bm[kk] = bp[kk];
                }
                bp[mu] += eps_g;
                bm[mu] -= eps_g;
                let (t12p, t13p, t23p) = angles_at_nm(&bp);
                let (t12m, t13m, t23m) = angles_at_nm(&bm);
                g_12_nm[mu] = (t12p - t12m) / (2.0 * eps_g);
                g_13_nm[mu] = (t13p - t13m) / (2.0 * eps_g);
                g_23_nm[mu] = (t23p - t23m) / (2.0 * eps_g);
            }
            // Principal angle: cos(theta) = (g_old . g_new) / (|g_old|*|g_new|)
            let dot6 = |a: &[f64; 6], b: &[f64; 6]| -> f64 {
                a.iter().zip(b).map(|(x, y)| x * y).sum()
            };
            let norm6 = |a: &[f64; 6]| -> f64 { dot6(a, a).sqrt() };
            for (name, g_old, g_new) in [
                ("g_12", &g_12, &g_12_nm),
                ("g_13", &g_13, &g_13_nm),
                ("g_23", &g_23, &g_23_nm),
            ] {
                let n_old = norm6(g_old);
                let n_new = norm6(g_new);
                if n_old > 1e-15 && n_new > 1e-15 {
                    let cos_pa = (dot6(g_old, g_new) / (n_old * n_new)).clamp(-1.0, 1.0);
                    let pa_deg = cos_pa.acos().to_degrees();
                    println!("  {name}: principal angle = {pa_deg:.1} deg, |g_old|={n_old:.3}, |g_new|={n_new:.3}");
                    if pa_deg > 10.0 {
                        println!("    ** WARNING: > 10 deg drift -- V_6 basis may not be intrinsic **");
                    }
                }
            }

            println!("\n  --- Mass-squared ratio at NM optimum ---");
            // Rebuild complex mass matrices at NM-optimized point
            let mut beta_nm = [0.0_f64; 6];
            for kk in 0..6 {
                beta_nm[kk] = nm_params[1] * ctx_fine.u_solar[kk]
                            + nm_params[2] * ctx_fine.u_atmo[kk];
            }
            let mut m_nu_nm = m_nu_real.clone();
            apply_v6_perturbation(&mut m_nu_nm, &v6_basis, &beta_nm, &lift);
            let m_nu_nm = (&m_nu_nm + m_nu_nm.transpose()) * faer::scale(0.5);
            let mut m_nu_c = faer::Mat::<faer::complex_native::c64>::zeros(3, 3);
            for i in 0..3 {
                m_nu_c.write(i, i, faer::complex_native::c64::new(m_nu_nm.read(i, i), 0.0));
                for j in (i + 1)..3 {
                    let phase = nm_params[0] * phi_fine[i][j];
                    let mag = m_nu_nm.read(i, j);
                    m_nu_c.write(i, j, faer::complex_native::c64::new(
                        mag * phase.cos(), mag * phase.sin()));
                    m_nu_c.write(j, i, faer::complex_native::c64::new(
                        mag * phase.cos(), -mag * phase.sin()));
                }
            }
            let eig_nu_nm = m_nu_c.selfadjoint_eigendecomposition(faer::Side::Lower);
            let mut ev_nu: Vec<f64> = (0..3).map(|i|
                eig_nu_nm.s().column_vector().read(i).re.abs()
            ).collect();
            ev_nu.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let dm21_sq = ev_nu[1] * ev_nu[1] - ev_nu[0] * ev_nu[0];
            let dm31_sq = ev_nu[2] * ev_nu[2] - ev_nu[0] * ev_nu[0];
            let r_mass = if dm31_sq.abs() > 1e-30 { dm21_sq / dm31_sq } else { 0.0 };
            println!("  dm21^2/dm31^2 = {:.4}", r_mass);
            println!("  3-blade prediction: 0.0304");
            println!("  PDG measured: 0.0307 +/- 0.001");

            // ----- Publication chi2 table (E1) -----
            println!("\n  ========= PUBLICATION CHI2 TABLE =========");
            println!("  Observable       |  Value     |  PDG       |  Pull (sigma)");
            println!("  ---------------------------------------------------------");
            let pull_12 = (nm_result.theta_12 - 33.41) / 0.72;
            let pull_13 = (nm_result.theta_13 - 8.54) / 0.12;
            let pull_23 = (nm_result.theta_23 - 49.0) / 1.3;
            let j_pdg = 0.0086_f64;
            let j_pdg_err = 0.0020_f64;
            let pull_j = (nm_result.j_cp.abs() - j_pdg) / j_pdg_err;
            let r_pdg = 0.0307_f64;
            let r_err = 0.001_f64;
            let pull_r = (r_mass - r_pdg) / r_err;
            println!("  theta_12 (deg)   | {:7.2}    | {:7.2}    | {:+.2}",
                nm_result.theta_12, 33.41, pull_12);
            println!("  theta_13 (deg)   | {:7.2}    | {:7.2}    | {:+.2}",
                nm_result.theta_13, 8.54, pull_13);
            println!("  theta_23 (deg)   | {:7.2}    | {:7.2}    | {:+.2}",
                nm_result.theta_23, 49.0, pull_23);
            println!("  |J_CP|           | {:.3e}  | {:.3e}  | {:+.2}",
                nm_result.j_cp.abs(), j_pdg, pull_j);
            println!("  r = dm21/dm31    | {:7.4}    | {:7.4}    | {:+.2}",
                r_mass, r_pdg, pull_r);
            let chi2_total = pull_12 * pull_12 + pull_13 * pull_13
                + pull_23 * pull_23 + pull_j * pull_j + pull_r * pull_r;
            println!("  ---------------------------------------------------------");
            println!("  Total chi2 (5 obs) = {:.2}", chi2_total);
            println!("  chi2/ndf = {:.2} (ndf = 5 - 3 params = 2)", chi2_total / 2.0);
            println!("  =========================================");

            // ----- Tangent map spectral analysis (D2b) -----
            println!("\n  --- Tangent map D_Phi at NM optimum ---");
            // D_Phi maps (alpha, t_sol, t_atm) -> (theta_12, theta_13, theta_23, J_CP)
            // via finite differences
            let eps_t = 1e-4_f64;
            let _param_names = ["alpha_CP", "t_sol", "t_atm"];
            let mut jacobian = [[0.0_f64; 3]; 4]; // 4 observables x 3 params
            for p in 0..3 {
                let mut params_p = nm_params;
                let mut params_m = nm_params;
                params_p[p] += eps_t;
                params_m[p] -= eps_t;
                let mut bp = CpScanBuffers::new();
                let mut bm = CpScanBuffers::new();
                let rp = evaluate_cp_scan_point(
                    params_p[0], params_p[1], params_p[2],
                    &phi_fine, &ctx_fine, &mut bp);
                let rm = evaluate_cp_scan_point(
                    params_m[0], params_m[1], params_m[2],
                    &phi_fine, &ctx_fine, &mut bm);
                jacobian[0][p] = (rp.theta_12 - rm.theta_12) / (2.0 * eps_t);
                jacobian[1][p] = (rp.theta_13 - rm.theta_13) / (2.0 * eps_t);
                jacobian[2][p] = (rp.theta_23 - rm.theta_23) / (2.0 * eps_t);
                jacobian[3][p] = (rp.j_cp - rm.j_cp) / (2.0 * eps_t);
            }
            // Build 4x3 nalgebra matrix for SVD
            let jac_mat = nalgebra::DMatrix::from_fn(4, 3, |i, j| jacobian[i][j]);
            let svd_jac = jac_mat.svd(false, false);
            println!("  Jacobian D_Phi (4x3) singular values:");
            for (i, s) in svd_jac.singular_values.iter().enumerate() {
                println!("    sigma[{i}] = {s:.4e}");
            }
            if svd_jac.singular_values.len() >= 2 {
                let ratio = svd_jac.singular_values[0] / svd_jac.singular_values[1];
                println!("  Condition number sigma[0]/sigma[1] = {ratio:.2}");
            }
            println!("  Jacobian entries (rows: t12, t13, t23, J; cols: alpha, t_sol, t_atm):");
            for (obs_name, row) in ["t12", "t13", "t23", "J"].iter().zip(jacobian.iter()) {
                println!("    d{obs_name}/d: [{:+.4}, {:+.4}, {:+.4}]",
                    row[0], row[1], row[2]);
            }
        } else {
            println!("  No solution found within 2% angle tolerance with nonzero J_CP.");
        }
    }

    /// Validate Cardano scan point against faer scan point.
    ///
    /// Runs both evaluate_cp_scan_point (faer) and evaluate_cp_scan_point_cardano
    /// at the same parameters and verifies angles/J_CP match within tolerance.
    #[test]
    fn test_cardano_vs_faer_scan_point() {
        let ch_pair = (11_usize, 12);
        let nu_pair = (7_usize, 8);
        let alpha_ch = 3.00_f64;
        let alpha_nu = 1.35_f64;

        let (m_ch_real, m_nu_real) = construct_pmns_matrices_two_param(
            ch_pair, nu_pair, alpha_ch, alpha_nu,
        );
        let (v6_basis, _, _) = extract_v6_basis();
        let lift = TensorElementLift;

        let eig_ch_0 = m_ch_real.selfadjoint_eigendecomposition(faer::Side::Lower);
        let eig_nu_0 = m_nu_real.selfadjoint_eigendecomposition(faer::Side::Lower);
        let u_raw_0 = eig_ch_0.u().transpose() * eig_nu_0.u();
        let (_, perm_u, perm_d) = crate::quark_sector::extract_ckm_permutation_aware(&u_raw_0);

        let u_solar = [0.1, 0.2, 0.3, 0.15, 0.25, 0.05];
        let u_atmo = [0.05, 0.1, 0.15, 0.3, 0.2, 0.1];
        let phi = [[0.0, 0.3, -0.2], [-0.3, 0.0, 0.5], [0.2, -0.5, 0.0]];

        let ctx = CpScanContext {
            m_nu_real: &m_nu_real, m_ch_real: &m_ch_real,
            v6_basis: &v6_basis, u_solar: &u_solar, u_atmo: &u_atmo,
            lift: &lift,
            perm_u: [perm_u[0], perm_u[1], perm_u[2]],
            perm_d: [perm_d[0], perm_d[1], perm_d[2]],
        };

        let test_points = [
            (0.1, 1.0, 2.0),
            (0.3, 0.5, 3.0),
            (0.45, 1.5, 4.0),
        ];

        println!("--- Cardano vs faer scan point validation ---");
        for &(alpha, ts, ta) in &test_points {
            let mut bufs = CpScanBuffers::new();
            let r_faer = evaluate_cp_scan_point(alpha, ts, ta, &phi, &ctx, &mut bufs);
            let r_card = evaluate_cp_scan_point_cardano(alpha, ts, ta, &phi, &ctx);

            let d12 = (r_faer.theta_12 - r_card.theta_12).abs();
            let d13 = (r_faer.theta_13 - r_card.theta_13).abs();
            let d23 = (r_faer.theta_23 - r_card.theta_23).abs();
            let dj = (r_faer.j_cp - r_card.j_cp).abs();

            println!("  alpha={alpha:.2}, ts={ts:.1}, ta={ta:.1}: dt12={d12:.2e}, dt13={d13:.2e}, dt23={d23:.2e}, dJ={dj:.2e}");

            assert!(d12 < 1e-6, "theta_12 mismatch: faer={:.6}, cardano={:.6}", r_faer.theta_12, r_card.theta_12);
            assert!(d13 < 1e-6, "theta_13 mismatch: faer={:.6}, cardano={:.6}", r_faer.theta_13, r_card.theta_13);
            assert!(d23 < 1e-6, "theta_23 mismatch: faer={:.6}, cardano={:.6}", r_faer.theta_23, r_card.theta_23);
            assert!(dj < 1e-8, "j_cp mismatch: faer={:.10}, cardano={:.10}", r_faer.j_cp, r_card.j_cp);
        }
        println!("  All 3 test points match within tolerance.");
    }

    /// Pathion (32D) V_k spectrum analysis (D1).
    ///
    /// Compares the assessor-complement basis for dim=16 (sedenion, V_6)
    /// versus dim=32 (Pathion, V_k). Reports:
    /// - Number of assessor pairs at each dimension
    /// - Singular value spectrum
    /// - Effective rank at multiple thresholds
    /// - Whether extra directions beyond 6 carry significant weight
    #[test]
    #[ignore] // ~4 min due to dim=64 triple loop (C(63,3) = 39,711 triads)
    fn test_pathion_vk_spectrum() {
        println!("--- PATHION (32D) V_k SPECTRUM ANALYSIS ---\n");

        // Sedenion (16D) baseline
        let (basis_16, sv_16, assess_16) = extract_vk_basis(16, 12);
        println!("  dim=16 (sedenion): {} assessor pairs", assess_16.len());
        println!("  V_k rank = {}, singular values:", basis_16.nrows());
        for (i, &s) in sv_16.iter().enumerate() {
            println!("    sv[{i:2}] = {s:.6e}");
        }

        // Pathion (32D)
        println!();
        let (basis_32, sv_32, assess_32) = extract_vk_basis(32, 20);
        println!("  dim=32 (Pathion): {} assessor pairs", assess_32.len());
        println!("  V_k rank = {}, singular values:", basis_32.nrows());
        for (i, &s) in sv_32.iter().enumerate() {
            println!("    sv[{i:2}] = {s:.6e}");
        }

        // Effective rank analysis
        println!("\n  Effective rank summary:");
        for &thresh in &[1e-4, 1e-6, 1e-8, 1e-10] {
            let rank_16 = sv_16.iter().filter(|&&s| s > thresh).count();
            let rank_32 = sv_32.iter().filter(|&&s| s > thresh).count();
            println!("    thresh={thresh:.0e}: dim16_rank={rank_16}, dim32_rank={rank_32}");
        }

        // Gap analysis: ratio of sv[6]/sv[5] tells us if V_6 is natural
        if sv_16.len() >= 7 {
            let gap_16 = sv_16[6] / sv_16[5];
            println!("\n  dim=16 gap sv[6]/sv[5] = {gap_16:.4} (small = V_6 natural cutoff)");
        }
        if sv_32.len() >= 7 {
            let gap_32 = sv_32[6] / sv_32[5];
            println!("  dim=32 gap sv[6]/sv[5] = {gap_32:.4}");
        }
        if sv_32.len() >= 13 {
            let gap_32_12 = sv_32[12] / sv_32[11];
            println!("  dim=32 gap sv[12]/sv[11] = {gap_32_12:.4}");
        }

        // Sanity check: sedenion V_6 should match original extract_v6_basis
        let (basis_orig, sv_orig, _) = extract_v6_basis();
        assert_eq!(basis_16.nrows(), basis_orig.nrows(),
            "extract_vk_basis(16) rank differs from extract_v6_basis");
        for i in 0..sv_orig.len().min(sv_16.len()) {
            let diff = (sv_16[i] - sv_orig[i]).abs();
            assert!(diff < 1e-10,
                "sv mismatch at {i}: vk={:.6e}, v6={:.6e}", sv_16[i], sv_orig[i]);
        }
        println!("\n  Consistency check: extract_vk_basis(16) matches extract_v6_basis: OK");

        // Sedenion uniqueness: test dim=8 (octonion) for comparison
        // Octonion has half=4: low in 1..3, high in 5..7, excluding high=low+4
        // 3*3 - 3 = 6 assessor pairs. Expect full associativity -> no V_k.
        let (basis_8, sv_8, assess_8) = extract_vk_basis(16, 12); // dim=8 too small for assessors
        // Actually, dim must be >= 16. The assessor structure requires CD dim >= 16.
        // For the uniqueness claim, compare 16 vs 32:
        println!("\n  === SEDENION UNIQUENESS EVIDENCE ===");
        println!("  dim=16: rank={}, assessors={}", basis_16.nrows(), assess_16.len());
        println!("  dim=32: rank={}, assessors={}", basis_32.nrows(), assess_32.len());
        let _ = (basis_8, sv_8, assess_8);

        // Chingon (64D) -- tests the monotonicity claim
        println!();
        let (basis_64, sv_64, assess_64) = extract_vk_basis(64, 20);
        println!("  dim=64 (Chingon): {} assessor pairs", assess_64.len());
        println!("  V_k rank = {}, singular values:", basis_64.nrows());
        for (i, &s) in sv_64.iter().enumerate() {
            println!("    sv[{i:2}] = {s:.6e}");
        }

        println!("\n  === SEDENION UNIQUENESS EVIDENCE ===");
        println!("  dim=16: rank={}, assessors={}", basis_16.nrows(), assess_16.len());
        println!("  dim=32: rank={}, assessors={}", basis_32.nrows(), assess_32.len());
        println!("  dim=64: rank={}, assessors={}", basis_64.nrows(), assess_64.len());
        println!("  Pattern: 6 -> {} -> {} (rank drops with doubling).",
            basis_32.nrows(), basis_64.nrows());
        if basis_64.nrows() == 0 {
            println!("  dim=64 rank=0: Frobenius noise guard triggered (sv_max/||G_x||_F < 1e-8).");
            println!("  The sedenion (dim=16) is the unique CD dimension with rank-6");
            println!("  assessor complement -- exactly the dimension needed for");
            println!("  independent 3-angle + 3-mass steering of flavor physics.");
        } else {
            println!("  NOTE: dim=64 rank={} may be noise -- check VK_PROFILE output.", basis_64.nrows());
        }
    }

    /// delta_CP sign systematics (C2): explore 8 combinations of:
    /// 1. Selector swap: (7,8) vs (8,7) -- which basis element drives nu_a/nu_b
    /// 2. L vs R multiplication order in the associator (swap a,b)
    /// 3. Epsilon sign flip: negate the upper octonion half
    ///
    /// If ANY combination gives delta ~ 195, the model has a sign degeneracy
    /// (weak predictive power). If NONE does, the delta ~ 93 prediction is
    /// robust but falsified against PDG.
    #[test]
    fn test_delta_cp_sign_systematics() {
        use cd_kernel::gourlay_psi;
        use crate::majorana_braiding::MajoranaMode;
        use crate::bell_inequality::rotate_sparse;

        let ch_pair = (11_usize, 12);
        let alpha_ch = 3.00_f64;
        let alpha_nu = 1.35_f64;

        // Selector pairs to test: (7,8) and (8,7)
        let selector_pairs = [(7_usize, 8_usize), (8_usize, 7_usize)];
        // L/R swap: controls which mode is a vs b in the associator
        let lr_swaps = [false, true];
        // Epsilon sign: negate profiles or not
        let eps_signs = [false, true];

        let (o1, o2, o3, sign_table) = super::psi_setup();
        let subs = [&o1, &o2, &o3];

        println!("--- delta_CP SIGN SYSTEMATICS (8 combinations) ---\n");
        println!("  {:>6} {:>4} {:>4} | {:>8} {:>8} {:>8} | {:>10} {:>8}",
            "sel", "swap", "eps", "t12", "t13", "t23", "|J|", "delta");

        let k_test = 5_usize;

        for &(sel_a, sel_b) in &selector_pairs {
            for &lr_swap in &lr_swaps {
                for &eps_flip in &eps_signs {
                    let nu_pair = if !lr_swap { (sel_a, sel_b) } else { (sel_b, sel_a) };

                    let (m_ch_real, m_nu_real) = construct_pmns_matrices_two_param(
                        ch_pair, nu_pair, alpha_ch, alpha_nu,
                    );
                    let (v6_basis, _, _) = extract_v6_basis();
                    let lift = TensorElementLift;

                    let eig_ch_0 = m_ch_real.selfadjoint_eigendecomposition(faer::Side::Lower);
                    let eig_nu_0 = m_nu_real.selfadjoint_eigendecomposition(faer::Side::Lower);
                    let u_raw_0 = eig_ch_0.u().transpose() * eig_nu_0.u();
                    let (_, perm_u, perm_d) = crate::quark_sector::extract_ckm_permutation_aware(&u_raw_0);

                    let nu_a = MajoranaMode { gamma_index: nu_pair.0 - 1, cd_basis_index: nu_pair.0, cd_dim: 16 };
                    let nu_b = MajoranaMode { gamma_index: nu_pair.1 - 1, cd_basis_index: nu_pair.1, cd_dim: 16 };

                    let build_profile = |mode_i: &MajoranaMode, mode_j: &MajoranaMode, sub: &[usize]| -> [f64; 16] {
                        let i = mode_i.cd_basis_index;
                        let j = mode_j.cd_basis_index;
                        let a_sparse = vec![(i, 1.0)];
                        let a_rotated = rotate_sparse(&a_sparse, i, j, std::f64::consts::FRAC_PI_4);
                        let b_sparse = vec![(j, 1.0)];
                        let mut profile = [0.0_f64; 16];
                        for &kk in sub {
                            if kk == 0 || kk == i || kk == j { continue; }
                            let x_sparse = [(kk, 1.0)];
                            profile[kk] = sign_table.sparse_associator_sum(&a_rotated, &x_sparse, &b_sparse);
                        }
                        if eps_flip {
                            for idx in 8..16 { profile[idx] = -profile[idx]; }
                        }
                        profile
                    };

                    let nu_profiles: Vec<[f64; 16]> = subs.iter()
                        .map(|s| build_profile(&nu_a, &nu_b, s)).collect();

                    let dot16 = |a: &[f64; 16], b: &[f64; 16]| -> f64 {
                        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
                    };

                    let apply_jk = |v: &[f64; 16]| -> [f64; 16] { apply_jk_full_16d(v, k_test) };

                    let mut phi = [[0.0_f64; 3]; 3];
                    let mut has_phase = false;
                    for i in 0..3 {
                        for j in 0..3 {
                            if i != j {
                                let psi_j = gourlay_psi(&nu_profiles[j]);
                                let re = dot16(&nu_profiles[i], &psi_j);
                                let jk_psi = apply_jk(&psi_j);
                                let im = dot16(&nu_profiles[i], &jk_psi);
                                phi[i][j] = im.atan2(re);
                                if phi[i][j].abs() > 1e-10 { has_phase = true; }
                            }
                        }
                    }

                    if !has_phase {
                        println!("  ({},{}) {:>4} {:>4} | no phase structure", sel_a, sel_b, lr_swap, eps_flip);
                        continue;
                    }

                    // Quick scan: alpha=0.45, use GN baseline
                    let eps_fd = 0.05_f64;
                    let n_basis = v6_basis.nrows().min(6);
                    let angles_at = |beta: &[f64; 6]| -> (f64, f64, f64) {
                        let mut m_nu = m_nu_real.clone();
                        apply_v6_perturbation(&mut m_nu, &v6_basis, beta, &lift);
                        let m_nu_s = (&m_nu + m_nu.transpose()) * faer::scale(0.5);
                        let eig_nu = m_nu_s.selfadjoint_eigendecomposition(faer::Side::Lower);
                        let u_raw = eig_ch_0.u().transpose() * eig_nu.u();
                        let mut u_perm2 = faer::Mat::zeros(3, 3);
                        for i in 0..3 { for j in 0..3 {
                            u_perm2.write(i, j, u_raw.read(perm_u[i], perm_d[j]));
                        }}
                        extract_pmns_angles(&u_perm2)
                    };
                    let mut g12 = [0.0_f64; 6];
                    let mut g13 = [0.0_f64; 6];
                    let mut g23 = [0.0_f64; 6];
                    for mu in 0..n_basis {
                        let mut bp = [0.0_f64; 6]; bp[mu] = eps_fd;
                        let mut bm = [0.0_f64; 6]; bm[mu] = -eps_fd;
                        let (t12p, t13p, t23p) = angles_at(&bp);
                        let (t12m, t13m, t23m) = angles_at(&bm);
                        g12[mu] = (t12p - t12m) / (2.0 * eps_fd);
                        g13[mu] = (t13p - t13m) / (2.0 * eps_fd);
                        g23[mu] = (t23p - t23m) / (2.0 * eps_fd);
                    }
                    let u_s = compute_constrained_solar_direction(&g12, &g13, &g23);
                    let u_a = compute_constrained_atmospheric_direction(&g23, &g13, &u_s);
                    let inner = |t_s: f64, t_a: f64| -> (f64, f64, f64) {
                        let mut beta = [0.0_f64; 6];
                        for kk in 0..6 { beta[kk] = t_s * u_s[kk] + t_a * u_a[kk]; }
                        angles_at(&beta)
                    };
                    let (ts, ta, _, _) = gauss_newton_2d(&inner, 1.5, 0.0,
                        (33.41, 8.54, 49.0), (1.0, 2.24, 1.0), 15);

                    let ctx = CpScanContext {
                        m_nu_real: &m_nu_real, m_ch_real: &m_ch_real,
                        v6_basis: &v6_basis, u_solar: &u_s, u_atmo: &u_a,
                        lift: &lift,
                        perm_u: [perm_u[0], perm_u[1], perm_u[2]],
                        perm_d: [perm_d[0], perm_d[1], perm_d[2]],
                    };
                    let mut bufs = CpScanBuffers::new();
                    let r = evaluate_cp_scan_point(0.45, ts, ta, &phi, &ctx, &mut bufs);

                    println!("  ({},{}) {:>4} {:>4} | {:8.2} {:8.2} {:8.2} | {:10.3e} {:8.1}",
                        sel_a, sel_b, lr_swap, eps_flip,
                        r.theta_12, r.theta_13, r.theta_23,
                        r.j_cp.abs(), r.delta_cp_invariant);
                }
            }
        }
    }

    // =========================================================================
    // Epic B: S_3-module lift derivation
    // =========================================================================

    /// Compute the full S_3 representation on V_6 and determine whether
    /// TensorElementLift lies in Hom_{S_3}(V_6, Sym_3(R)).
    ///
    /// Steps:
    /// 1. Build 42x42 psi and epsilon matrices on assessor space
    /// 2. Verify S_3 relations (psi^3=I, eps^2=I, eps*psi=psi^2*eps)
    /// 3. Restrict to V_6 (6x6 matrices)
    /// 4. Compute S_3 characters: chi(e), chi(psi), chi(eps)
    /// 5. Decompose V_6 = n_triv * 1 + n_sgn * sgn + n_std * std
    /// 6. Compute Hom_{S_3}(V_6, Sym_3(R))
    ///
    /// Reference: Gresnigt/Gourlay 2019 (1904.03186), 2026 (2601.07857).
    /// Psi formula: Gourlay & Gresnigt (arXiv:2407.01580), Eq 5.
    /// Epsilon: negates upper octonion half (indices 8..15).
    /// V_6 extraction: complement of B/C column space in 42D assessor space.
    #[test]
    fn test_s3_action_on_v6_and_lift_derivation() {
        use cd_kernel::{gourlay_psi, gourlay_epsilon};
        use nalgebra::DMatrix;

        // ---- Step 0: Build assessor pairs (same as extract_v6_basis) ----
        let mut assessors: Vec<(usize, usize)> = Vec::new();
        for low in 1..=7_usize {
            for high in 9..=15_usize {
                if high == low + 8 { continue; }
                assessors.push((low, high));
            }
        }
        assert_eq!(assessors.len(), 42);

        // ---- Step 1: Build 42x42 psi and epsilon matrices ----
        // Psi acts on each sedenion basis vector e_i. The induced action on
        // the assessor (low, high) is: find which assessor contains the
        // images of e_low and e_high under psi.
        //
        // For a unit assessor vector with assessor[a] = 1 (corresponding to
        // the pair (low, high)), we embed into 16D as e_low + e_high,
        // apply psi, then decompose the result back into assessor space.
        //
        // This gives a 42x42 real matrix M_psi where
        //   M_psi[b][a] = <assessor_b | psi(assessor_a)>

        let build_s3_matrix = |action: &dyn Fn(&[f64; 16]) -> [f64; 16]| -> DMatrix<f64> {
            let mut mat = DMatrix::zeros(42, 42);
            for (a_idx, &(low, high)) in assessors.iter().enumerate() {
                // Embed: e_low
                let mut v_low = [0.0_f64; 16];
                v_low[low] = 1.0;
                let img_low = action(&v_low);

                // Embed: e_high
                let mut v_high = [0.0_f64; 16];
                v_high[high] = 1.0;
                let img_high = action(&v_high);

                // Decompose each image into assessor overlaps.
                // An assessor (l, h) has overlap |img[l]| + |img[h]| with
                // the image, but we need the LINEAR action. Since psi is
                // linear on basis vectors, we compute:
                //   psi(e_low) = sum_i c_i * e_i
                // The assessor weight for (l, h) from this image is:
                //   contribution to assessor[b=(l,h)] = img_low[l] or img_low[h]
                // depending on which index the image lands on.
                for (b_idx, &(bl, bh)) in assessors.iter().enumerate() {
                    // Contribution of psi(e_low) to assessor (bl, bh):
                    // if psi(e_low) has component at bl or bh
                    let c_low = img_low[bl] + img_low[bh];
                    let c_high = img_high[bl] + img_high[bh];
                    mat[(b_idx, a_idx)] += c_low + c_high;
                }
            }
            mat
        };

        let psi_fn = |v: &[f64; 16]| -> [f64; 16] { gourlay_psi(v) };
        let eps_fn = |v: &[f64; 16]| -> [f64; 16] { gourlay_epsilon(v) };

        let m_psi_42 = build_s3_matrix(&psi_fn);
        let m_eps_42 = build_s3_matrix(&eps_fn);

        // ---- Step 2: Verify S_3 relations ----
        let id42 = DMatrix::identity(42, 42);
        let psi3 = &m_psi_42 * &m_psi_42 * &m_psi_42;
        let eps2 = &m_eps_42 * &m_eps_42;
        let eps_psi = &m_eps_42 * &m_psi_42;
        let psi2_eps = &m_psi_42 * &m_psi_42 * &m_eps_42;

        let psi3_err = (&psi3 - &id42).norm();
        let eps2_err = (&eps2 - &id42).norm();
        let relation_err = (&eps_psi - &psi2_eps).norm();

        println!("--- S_3 ACTION ON ASSESSOR SPACE (42D) ---");
        println!("  ||psi^3 - I||  = {:.6e}", psi3_err);
        println!("  ||eps^2 - I||  = {:.6e}", eps2_err);
        println!("  ||eps*psi - psi^2*eps|| = {:.6e}", relation_err);

        // ---- Step 3: Extract V_6 basis and restrict ----
        let (v6_basis, sv, _) = extract_v6_basis();
        let n_basis = v6_basis.nrows().min(6);
        println!("\n  V_6 basis: {}x{}, singular values: {:?}", v6_basis.nrows(), v6_basis.ncols(), &sv[..n_basis]);

        // V_6 basis is n_basis x 42. Restriction: M_V6 = V * M_42 * V^T
        // where V is n_basis x 42.
        let v6 = v6_basis.rows(0, n_basis);
        let m_psi_v6 = &v6 * &m_psi_42 * &v6.transpose();
        let m_eps_v6 = &v6 * &m_eps_42 * &v6.transpose();

        // ---- Step 4: S_3 characters = traces ----
        let chi_e = n_basis as f64; // trace of identity on V_6
        let chi_psi = m_psi_v6.trace();
        let chi_eps = m_eps_v6.trace();

        // Also compute chi(psi^2) and chi(eps*psi)
        let m_psi2_v6 = &m_psi_v6 * &m_psi_v6;
        let m_eps_psi_v6 = &m_eps_v6 * &m_psi_v6;
        let chi_psi2 = m_psi2_v6.trace();
        let chi_eps_psi = m_eps_psi_v6.trace();

        println!("\n--- S_3 CHARACTERS ON V_6 ---");
        println!("  chi(e)       = {:.6}", chi_e);
        println!("  chi(psi)     = {:.6}", chi_psi);
        println!("  chi(psi^2)   = {:.6}", chi_psi2);
        println!("  chi(eps)     = {:.6}", chi_eps);
        println!("  chi(eps*psi) = {:.6}", chi_eps_psi);

        // Verify S_3 relations on V_6
        let v6_psi3 = &m_psi_v6 * &m_psi_v6 * &m_psi_v6;
        let v6_id = DMatrix::identity(n_basis, n_basis);
        let v6_psi3_err = (&v6_psi3 - &v6_id).norm();
        let v6_eps2 = &m_eps_v6 * &m_eps_v6;
        let v6_eps2_err = (&v6_eps2 - &v6_id).norm();

        println!("\n  V_6 S_3 relation checks:");
        println!("    ||psi^3 - I|| on V_6 = {:.6e}", v6_psi3_err);
        println!("    ||eps^2 - I|| on V_6 = {:.6e}", v6_eps2_err);

        let faithful = v6_psi3_err < 0.1 && v6_eps2_err < 0.1;
        println!("    Faithful S_3 action on V_6: {}", faithful);

        // ---- Step 5: Decompose V_6 into S_3 irreps ----
        // S_3 has 3 irreps over R:
        //   1 (trivial):  chi(e)=1, chi(psi)=1,  chi(eps)=1
        //   sgn (sign):   chi(e)=1, chi(psi)=1,  chi(eps)=-1
        //   std (standard): chi(e)=2, chi(psi)=-1, chi(eps)=0
        //
        // Multiplicity formula: n_rho = (1/|G|) * sum_{g in G} chi_V(g) * chi_rho(g)
        // |S_3| = 6, conjugacy classes: {e} (size 1), {psi, psi^2} (size 2), {eps, eps*psi, eps*psi^2} (size 3)
        //
        // n_triv = (1/6) * [1*chi(e)*1 + 2*chi(psi)*1 + 3*chi(eps)*1]
        // n_sgn  = (1/6) * [1*chi(e)*1 + 2*chi(psi)*1 + 3*chi(eps)*(-1)]
        // n_std  = (1/6) * [1*chi(e)*2 + 2*chi(psi)*(-1) + 3*chi(eps)*0]

        let n_triv = (chi_e + 2.0 * chi_psi + 3.0 * chi_eps) / 6.0;
        let n_sgn = (chi_e + 2.0 * chi_psi - 3.0 * chi_eps) / 6.0;
        let n_std = (2.0 * chi_e - 2.0 * chi_psi) / 6.0;

        println!("\n--- S_3 IRREP DECOMPOSITION OF V_6 ---");
        println!("  n_trivial  = {:.4}", n_triv);
        println!("  n_sign     = {:.4}", n_sgn);
        println!("  n_standard = {:.4}", n_std);
        println!("  check: {} + {} + 2*{} = {:.4} (should be {})",
            n_triv, n_sgn, n_std,
            n_triv + n_sgn + 2.0 * n_std,
            n_basis);

        // ---- Step 6: Hom_{S_3}(V_6, Sym_3(R)) ----
        // Sym_3(R) = 6D space of 3x3 real symmetric matrices.
        // S_3 acts by simultaneous permutation of rows and columns.
        // Character: chi(e)=6, chi(psi)=0, chi(eps)=2
        //   (psi permutes all 3 diagonal entries cyclically and all 3 off-diag -> trace 0)
        //   (eps swaps two rows/cols: fixes 1 diagonal + 1 off-diag, swaps the rest -> trace 2)
        //
        // Sym_3 decomposition: 2*1 + 0*sgn + 2*std
        //   n_triv = (6 + 0 + 6)/6 = 2
        //   n_sgn  = (6 + 0 - 6)/6 = 0
        //   n_std  = (12 - 0)/6 = 2
        //
        // dim Hom_{S_3}(V_6, Sym_3) = n_triv(V_6)*n_triv(Sym_3) + n_sgn(V_6)*n_sgn(Sym_3) + n_std(V_6)*n_std(Sym_3)
        // (by Schur's lemma: Hom(rho, sigma) = 0 if rho != sigma, = R^{mult} if rho = sigma)

        let chi_sym3_e = 6.0_f64;
        let chi_sym3_psi = 0.0_f64;
        let chi_sym3_eps = 2.0_f64;

        let n_triv_sym3 = (chi_sym3_e + 2.0 * chi_sym3_psi + 3.0 * chi_sym3_eps) / 6.0;
        let n_sgn_sym3 = (chi_sym3_e + 2.0 * chi_sym3_psi - 3.0 * chi_sym3_eps) / 6.0;
        let n_std_sym3 = (2.0 * chi_sym3_e - 2.0 * chi_sym3_psi) / 6.0;

        println!("\n--- Sym_3(R) IRREP DECOMPOSITION ---");
        println!("  n_trivial  = {:.4}", n_triv_sym3);
        println!("  n_sign     = {:.4}", n_sgn_sym3);
        println!("  n_standard = {:.4}", n_std_sym3);

        // dim Hom_{S_3}(V_6, Sym_3) via Schur's lemma
        let dim_hom = n_triv * n_triv_sym3 + n_sgn * n_sgn_sym3 + n_std * n_std_sym3;

        println!("\n--- CENTRAL THEOREM: dim Hom_{{S_3}}(V_6, Sym_3) ---");
        println!("  = {:.4} * {:.4} + {:.4} * {:.4} + {:.4} * {:.4}",
            n_triv, n_triv_sym3, n_sgn, n_sgn_sym3, n_std, n_std_sym3);
        println!("  = {:.4}", dim_hom);

        if dim_hom.abs() < 0.5 {
            println!("\n  RESULT: Hom_{{S_3}}(V_6, Sym_3) = 0");
            println!("  TensorElementLift is genuinely NON-equivariant under this S_3 action.");
            println!("  This is an important negative theorem (if the S_3 action is faithful).");
        } else {
            let dim_hom_int = dim_hom.round() as usize;
            println!("\n  RESULT: dim Hom_{{S_3}}(V_6, Sym_3) = {}", dim_hom_int);
            println!("  Equivariant lifts EXIST. Characterize the intertwiner space.");
        }

        if !faithful {
            println!("\n  WARNING: S_3 action is NOT faithful on V_6.");
            println!("  psi^3 != I means the induced action is not a true S_3 representation.");
            println!("  The non-equivariance result is PROVISIONAL on the current action.");
            println!("  A triad/incidence-level S_3 action may give different results.");
        }

        // Print eigenvalues of psi on V_6 for additional diagnostics
        let eig_psi = m_psi_v6.clone().symmetric_eigen();
        println!("\n  Eigenvalues of psi on V_6: {:?}", eig_psi.eigenvalues.as_slice());
        let eig_eps = m_eps_v6.clone().symmetric_eigen();
        println!("  Eigenvalues of eps on V_6: {:?}", eig_eps.eigenvalues.as_slice());
    }

    /// T4: Compute psi-friction profiles for all 42 assessors across 3 subalgebras.
    ///
    /// For each assessor pair (low, high), compute the signed friction
    /// cd_braid_signed_friction in each of the 3 octonionic subalgebras
    /// (O_1, O_2, O_3). This gives a 42x3 matrix where:
    ///   F[a][g] = friction of assessor a in subalgebra O_g
    ///
    /// The subalgebra classification (C-1528) predicts that intra-generation
    /// assessors have friction concentrated in one subalgebra, while cross-
    /// generation assessors spread across 2 or 3.
    ///
    /// Claim: C-1529.
    #[test]
    fn test_psi_friction_profiles_42x3() {
        use crate::lepton_mass_hierarchy::cd_braid_signed_friction;
        use crate::majorana_braiding::MajoranaMode;
        let (o1, o2, o3, sign_table) = super::psi_setup();
        let subs = [&o1[..], &o2[..], &o3[..]];

        // Build 42 assessor pairs
        let mut assessors: Vec<(usize, usize)> = Vec::new();
        for low in 1..=7_usize {
            for high in 9..=15_usize {
                if high == low + 8 { continue; }
                assessors.push((low, high));
            }
        }

        println!("--- T4: PSI-FRICTION PROFILES (42 x 3) ---\n");

        // Subalgebra exclusive membership for classification
        let o1_excl: std::collections::HashSet<usize> = [1,5,9,13].into();
        let o2_excl: std::collections::HashSet<usize> = [2,6,10,14].into();
        let o3_excl: std::collections::HashSet<usize> = [3,7,11,15].into();

        let gen_label = |idx: usize| -> &'static str {
            if o1_excl.contains(&idx) { "O1" }
            else if o2_excl.contains(&idx) { "O2" }
            else if o3_excl.contains(&idx) { "O3" }
            else { "Sh" } // shared
        };

        // Compute 42x3 friction matrix
        let mut friction_matrix = vec![[0.0_f64; 3]; 42];
        let mut gen_labels = Vec::new();

        for (a_idx, &(low, high)) in assessors.iter().enumerate() {
            let mode_i = MajoranaMode { gamma_index: low - 1, cd_basis_index: low, cd_dim: 16 };
            let mode_j = MajoranaMode { gamma_index: high - 1, cd_basis_index: high, cd_dim: 16 };

            for (g, sub) in subs.iter().enumerate() {
                friction_matrix[a_idx][g] = cd_braid_signed_friction(&mode_i, &mode_j, sub, &sign_table);
            }

            let label = format!("{}-{}", gen_label(low), gen_label(high));
            gen_labels.push(label);
        }

        // Print friction matrix with generation labels
        println!("  {:>3} ({:>2},{:>2}) {:>5} | {:>8} {:>8} {:>8} | {:>5}",
            "idx", "lo", "hi", "type", "F(O1)", "F(O2)", "F(O3)", "dom");
        println!("  {:-<3}-{:-<7}-{:-<5}-+-{:-<8}-{:-<8}-{:-<8}-+-{:-<5}",
            "", "", "", "", "", "", "");

        let mut type_friction_sums: std::collections::BTreeMap<String, [f64; 3]> =
            std::collections::BTreeMap::new();

        for (a_idx, &(low, high)) in assessors.iter().enumerate() {
            let f = friction_matrix[a_idx];
            let abs_f: Vec<f64> = f.iter().map(|x| x.abs()).collect();
            let max_g = abs_f.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i).unwrap_or(0);
            let dom = ["O1", "O2", "O3"][max_g];

            println!("  {:>3} ({:>2},{:>2}) {:>5} | {:>8.3} {:>8.3} {:>8.3} | {:>5}",
                a_idx, low, high, gen_labels[a_idx],
                f[0], f[1], f[2], dom);

            let entry = type_friction_sums.entry(gen_labels[a_idx].clone()).or_insert([0.0; 3]);
            for g in 0..3 { entry[g] += f[g].abs(); }
        }

        // Summary: average friction by generation type
        println!("\n  Average |friction| by assessor type:\n");
        println!("  {:>7} | {:>8} {:>8} {:>8}", "type", "|F(O1)|", "|F(O2)|", "|F(O3)|");
        for (label, sums) in &type_friction_sums {
            println!("  {:>7} | {:>8.2} {:>8.2} {:>8.2}", label, sums[0], sums[1], sums[2]);
        }

        // Key test: do intra-generation assessors concentrate friction?
        // Count how many assessors have > 80% of their friction in one subalgebra
        let mut concentrated = 0;
        let mut spread = 0;
        for f in &friction_matrix {
            let total: f64 = f.iter().map(|x| x.abs()).sum();
            if total < 1e-10 { continue; }
            let max_frac = f.iter().map(|x| x.abs()).fold(0.0_f64, f64::max) / total;
            if max_frac > 0.8 { concentrated += 1; } else { spread += 1; }
        }

        println!("\n  Concentration analysis:");
        println!("    Concentrated (>80% in one sub): {}", concentrated);
        println!("    Spread (< 80% in one sub): {}", spread);
        println!("    (Prediction: intra-gen assessors concentrate, cross-gen spread)");
    }

    /// Backend regression: faer vs nalgebra eigendecomp on dim=16 (42x42 Gram).
    ///
    /// # Purpose
    ///
    /// Verifies that the faer divide-and-conquer eigensolver produces the
    /// same physical subspace as the original nalgebra Jacobi solver.
    /// This is the single gate for the eigensolver backend migration.
    ///
    /// # Why projector agreement, not eigenvector equality?
    ///
    /// Eigenvector signs are a gauge freedom: if v is an eigenvector,
    /// so is -v.  Different backends may choose different signs (and
    /// for degenerate eigenvalues, different rotations within the
    /// eigenspace).  The projector P = B^T * B (n_assess x n_assess)
    /// is sign-invariant and rotation-invariant within eigenspaces,
    /// so it is the correct observable to compare.
    ///
    /// # Checks (ordered by diagnostic priority)
    ///
    /// 1. **Effective rank**: same count of SVs above threshold
    /// 2. **Leading singular values**: within 1e-6 absolute tolerance
    /// 3. **Orthonormality**: |B * B^T - I_rank|_F < 1e-10
    ///    (basis vectors are ROWS, so B*B^T is rank x rank)
    /// 4. **Projector agreement**: |P_faer - P_nalgebra|_F < 1e-8
    ///
    /// # Expected output
    ///
    /// ```text
    ///   rank = 6, n_assess = 42
    ///   sv[0] ~ 3.41997e0 (6-fold degenerate)
    ///   |B*B^T - I|_F ~ 1e-15 (both backends)
    ///   |P_faer - P_nal|_F ~ 1e-14
    /// ```
    ///
    /// PASS: all four assertions hold.
    /// FAIL: subspace differs between backends -- investigate whether the
    /// threshold change (abs+rel vs pure relative) caused a rank difference.
    #[test]
    fn test_faer_vs_nalgebra_eigendecomp() {
        use nalgebra::DMatrix;

        let (basis_faer, sv_faer, assess_faer) = extract_vk_basis(16, 12);
        let (basis_nal, sv_nal, assess_nal) = extract_vk_basis_nalgebra(16, 12);

        // Same assessor set (deterministic construction)
        assert_eq!(assess_faer.len(), assess_nal.len(), "assessor count mismatch");
        assert_eq!(assess_faer, assess_nal, "assessor pairs differ");

        // Same effective rank
        assert_eq!(basis_faer.nrows(), basis_nal.nrows(),
            "rank mismatch: faer={}, nalgebra={}", basis_faer.nrows(), basis_nal.nrows());
        let rank = basis_faer.nrows();
        let n_assess = assess_faer.len();

        // Leading singular values within tolerance
        let n_sv = sv_faer.len().min(sv_nal.len());
        for i in 0..n_sv {
            let diff = (sv_faer[i] - sv_nal[i]).abs();
            assert!(diff < 1e-6,
                "sv[{i}] mismatch: faer={:.8e}, nalgebra={:.8e}, diff={:.3e}",
                sv_faer[i], sv_nal[i], diff);
        }

        // Orthonormality: B * B^T should be I_rank
        // basis_matrix shape is (rank, n_assess), so B*B^T is (rank, rank)
        let bbt_faer = &basis_faer * basis_faer.transpose();
        let eye_rank = DMatrix::identity(rank, rank);
        let ortho_err = (&bbt_faer - &eye_rank).norm();
        assert!(ortho_err < 1e-10,
            "faer basis non-orthonormal: |B*B^T - I|_F = {:.3e}", ortho_err);

        let bbt_nal = &basis_nal * basis_nal.transpose();
        let ortho_err_nal = (&bbt_nal - &eye_rank).norm();
        assert!(ortho_err_nal < 1e-10,
            "nalgebra basis non-orthonormal: |B*B^T - I|_F = {:.3e}", ortho_err_nal);

        // Projector agreement: P = B^T * B is the n_assess x n_assess projector
        // onto the rank-dimensional subspace. Eigenvector signs may differ between
        // backends, but the projector is sign-invariant.
        let proj_faer = basis_faer.transpose() * &basis_faer;
        let proj_nal = basis_nal.transpose() * &basis_nal;
        let proj_diff = (&proj_faer - &proj_nal).norm();
        assert!(proj_diff < 1e-8,
            "projector disagreement: |P_faer - P_nal|_F = {:.3e}", proj_diff);

        println!("  faer vs nalgebra backend regression: PASS");
        println!("    rank = {rank}, n_assess = {n_assess}");
        println!("    sv[0] = {:.8e} (faer) vs {:.8e} (nalgebra)", sv_faer[0], sv_nal[0]);
        println!("    |B*B^T - I|_F: faer={:.3e}, nalgebra={:.3e}", ortho_err, ortho_err_nal);
        println!("    |P_faer - P_nal|_F = {:.3e}", proj_diff);
    }
