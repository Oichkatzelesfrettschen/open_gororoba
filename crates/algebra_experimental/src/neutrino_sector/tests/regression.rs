use super::super::*;

/// Gauss-Newton optimization of the two-selector-type model.
///
/// 3 parameters: (beta, alpha_ch, alpha_nu)
/// 4 residuals: (theta_12, theta_13, theta_23, r) vs PDG
///
/// Uses the same LM-damped Gauss-Newton as C-1492 but generalized
/// to 3 parameters and 4 observables.
#[test]
fn test_two_selector_gauss_newton() {
    use crate::{
        bell_inequality::{SignTableCache, rotate_sparse},
        lepton_mass_hierarchy::cd_braid_signed_friction,
        majorana_braiding::MajoranaMode,
        three_fermion_generations::get_sedenion_subalgebras,
    };
    use cd_kernel::gourlay_psi;
    use nalgebra::{DMatrix, DVector};

    let pdg = Pdg2024::default();
    let pdg_r = 0.0307_f64;
    let pdg_targets = [pdg.theta_12_deg, pdg.theta_13_deg, pdg.theta_23_deg, pdg_r];
    let pdg_sigma = [pdg.theta_12_err, pdg.theta_13_err, pdg.theta_23_err, 0.003]; // r sigma ~10%

    let (o1, o2, o3) = get_sedenion_subalgebras();
    let subs = [&o1, &o2, &o3];
    let sign_table = SignTableCache::new(16);
    let w1 = -0.656850_f64;
    let w2 = -0.741999_f64;

    // Precompute 3-blade scalar frictions
    let ch_triple = (1_usize, 6, 11);
    let nu_triple = (1_usize, 3, 8);
    let sel_ch_3: Vec<f64> = subs
        .iter()
        .map(|s| {
            let (a, b, c) = ch_triple;
            let ma = MajoranaMode {
                gamma_index: a - 1,
                cd_basis_index: a,
                cd_dim: 16,
            };
            let mb = MajoranaMode {
                gamma_index: b - 1,
                cd_basis_index: b,
                cd_dim: 16,
            };
            let mc = MajoranaMode {
                gamma_index: c - 1,
                cd_basis_index: c,
                cd_dim: 16,
            };
            cd_braid_signed_friction(&ma, &mb, s, &sign_table)
                + cd_braid_signed_friction(&ma, &mc, s, &sign_table)
                + cd_braid_signed_friction(&mb, &mc, s, &sign_table)
        })
        .collect();
    let sel_nu_3: Vec<f64> = subs
        .iter()
        .map(|s| {
            let (a, b, c) = nu_triple;
            let ma = MajoranaMode {
                gamma_index: a - 1,
                cd_basis_index: a,
                cd_dim: 16,
            };
            let mb = MajoranaMode {
                gamma_index: b - 1,
                cd_basis_index: b,
                cd_dim: 16,
            };
            let mc = MajoranaMode {
                gamma_index: c - 1,
                cd_basis_index: c,
                cd_dim: 16,
            };
            cd_braid_signed_friction(&ma, &mb, s, &sign_table)
                + cd_braid_signed_friction(&ma, &mc, s, &sign_table)
                + cd_braid_signed_friction(&mb, &mc, s, &sign_table)
        })
        .collect();

    // Precompute 2-blade profiles
    let build_profile = |sel: (usize, usize), sub: &[usize]| -> [f64; 16] {
        let i = sel.0;
        let j = sel.1;
        let a_sparse = vec![(i, 1.0)];
        let a_rotated = rotate_sparse(&a_sparse, i, j, std::f64::consts::FRAC_PI_4);
        let b_sparse = vec![(j, 1.0)];
        let mut profile = [0.0_f64; 16];
        for &k in sub {
            if k == 0 || k == i || k == j {
                continue;
            }
            let x_sparse = [(k, 1.0)];
            profile[k] = sign_table.sparse_associator_sum(&a_rotated, &x_sparse, &b_sparse);
        }
        profile
    };
    let ch_prof: Vec<[f64; 16]> = subs.iter().map(|s| build_profile((11, 12), s)).collect();
    let nu_prof: Vec<[f64; 16]> = subs.iter().map(|s| build_profile((7, 8), s)).collect();

    let dot16 =
        |a: &[f64; 16], b: &[f64; 16]| -> f64 { a.iter().zip(b.iter()).map(|(x, y)| x * y).sum() };

    // Evaluation function: (beta, alpha_ch, alpha_nu) -> (t12, t13, t23, r)
    let evaluate = |beta: f64, a_ch: f64, a_nu: f64| -> [f64; 4] {
        let cb =
            construct_casimir_baseline(crate::quark_sector::SubalgebraScheme::InterleavedStride);
        let (m_base_ch, m_base_nu) = assemble_lepton_baseline(&cb);
        let mut m_ch = m_base_ch;
        let mut m_nu = m_base_nu;

        for g in 0..3 {
            let f_ch = beta * (w1 * sel_ch_3[g] + w2 * sel_nu_3[g]);
            let f_nu = beta * (w1 * sel_nu_3[g] + w2 * sel_ch_3[g]);
            m_ch[(g, g)] = m_ch[(g, g)] + f_ch.exp();
            m_nu[(g, g)] = m_nu[(g, g)] + f_nu.exp();
        }
        for i in 0..3 {
            for j in 0..3 {
                if i == j {
                    continue;
                }
                let psi_nu_j = gourlay_psi(&nu_prof[j]);
                let psi_ch_j = gourlay_psi(&ch_prof[j]);
                m_nu[(i, j)] = m_nu[(i, j)] + a_nu * dot16(&nu_prof[i], &psi_nu_j);
                m_ch[(i, j)] = m_ch[(i, j)] + a_ch * dot16(&ch_prof[i], &psi_ch_j);
            }
        }
        let m_ch_s = (&m_ch + m_ch.transpose()) * faer::Scale(0.5);
        let m_nu_s = (&m_nu + m_nu.transpose()) * faer::Scale(0.5);
        let eig_ch = m_ch_s.self_adjoint_eigen(faer::Side::Lower).unwrap();
        let eig_nu = m_nu_s.self_adjoint_eigen(faer::Side::Lower).unwrap();
        let u_raw = eig_ch.U().transpose() * eig_nu.U();
        let (u_pmns, _, _) = crate::quark_sector::extract_ckm_permutation_aware(&u_raw);
        let (t12, t13, t23) = extract_pmns_angles(&u_pmns);
        let mut ev: Vec<f64> = (0..3)
            .map(|i| eig_nu.S().column_vector()[i].abs())
            .collect();
        ev.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let dm21 = ev[1] * ev[1] - ev[0] * ev[0];
        let dm31 = ev[2] * ev[2] - ev[0] * ev[0];
        let r = if dm31.abs() > 1e-30 { dm21 / dm31 } else { 1.0 };
        [t12, t13, t23, r]
    };

    // Gauss-Newton with LM damping
    let n_params = 3;
    let n_resid = 4;
    let eps = 0.005_f64;
    let mut params = [0.60_f64, 8.0, 1.5]; // initial from grid scan
    let max_iter = 100;

    println!("  === Gauss-Newton: Two-Selector-Type Model ===\n");
    println!(
        "  {:>4} | {:>6} {:>6} {:>6} | {:>8} {:>8} {:>8} {:>8} | {:>8}",
        "iter", "beta", "a_ch", "a_nu", "t12", "t13", "t23", "r", "cost"
    );

    for iter in 0..max_iter {
        let obs = evaluate(params[0], params[1], params[2]);
        let residuals: Vec<f64> = (0..n_resid)
            .map(|i| (obs[i] - pdg_targets[i]) / pdg_sigma[i])
            .collect();
        let cost: f64 = residuals.iter().map(|r| r * r).sum();

        if iter % 10 == 0 || iter < 5 {
            println!(
                "  {:>4} | {:>6.3} {:>6.2} {:>6.2} | {:>8.2} {:>8.2} {:>8.2} {:>8.4} | {:>8.2}",
                iter, params[0], params[1], params[2], obs[0], obs[1], obs[2], obs[3], cost
            );
        }

        // Jacobian via central differences
        let mut jac = DMatrix::zeros(n_resid, n_params);
        for p in 0..n_params {
            let mut pp = params;
            let mut pm = params;
            pp[p] += eps;
            pm[p] -= eps;
            let obs_p = evaluate(pp[0], pp[1], pp[2]);
            let obs_m = evaluate(pm[0], pm[1], pm[2]);
            for r in 0..n_resid {
                jac[(r, p)] = (obs_p[r] - obs_m[r]) / (2.0 * eps * pdg_sigma[r]);
            }
        }

        // Normal equations: (J^T J + lambda I) delta = -J^T r
        let jt = jac.transpose();
        let jtj = &jt * &jac;
        let jtr = &jt * DVector::from_row_slice(&residuals);

        let lambda = 0.1_f64;
        let mut jtj_damped = jtj.clone();
        for i in 0..n_params {
            jtj_damped[(i, i)] += lambda;
        }

        let delta = match jtj_damped.clone().lu().solve(&(-&jtr)) {
            Some(d) => d,
            None => break,
        };

        // Line search
        let mut step = 1.0_f64;
        for _ in 0..10 {
            let new_params = [
                (params[0] + step * delta[0]).max(0.01).min(2.0),
                (params[1] + step * delta[1]).max(0.0).min(20.0),
                (params[2] + step * delta[2]).max(0.0).min(10.0),
            ];
            let new_obs = evaluate(new_params[0], new_params[1], new_params[2]);
            let new_cost: f64 = (0..n_resid)
                .map(|i| ((new_obs[i] - pdg_targets[i]) / pdg_sigma[i]).powi(2))
                .sum();
            if new_cost < cost {
                params = new_params;
                break;
            }
            step *= 0.5;
        }

        if delta.norm() < 1e-6 {
            break;
        }
    }

    // Compute initial cost before multi-start
    let init_obs = evaluate(params[0], params[1], params[2]);
    let init_cost: f64 = (0..n_resid)
        .map(|i| ((init_obs[i] - pdg_targets[i]) / pdg_sigma[i]).powi(2))
        .sum();

    // Multi-start from different initial conditions to escape local minima
    let starts = [
        [0.60, 8.0, 1.5],
        [0.30, 4.0, 1.0],
        [0.10, 2.0, 0.5],
        [0.80, 6.0, 2.0],
        [0.15, 10.0, 0.8],
        [0.50, 3.0, 3.0],
    ];

    let mut global_best_params = params;
    let mut global_best_cost = init_cost;

    for start in &starts {
        let mut p = *start;
        for _iter in 0..50 {
            let obs = evaluate(p[0], p[1], p[2]);
            let res: Vec<f64> = (0..n_resid)
                .map(|i| (obs[i] - pdg_targets[i]) / pdg_sigma[i])
                .collect();
            let cost: f64 = res.iter().map(|r| r * r).sum();

            let mut jac = DMatrix::zeros(n_resid, n_params);
            for pi in 0..n_params {
                let mut pp = p;
                let mut pm = p;
                pp[pi] += eps;
                pm[pi] -= eps;
                let obs_p = evaluate(pp[0], pp[1], pp[2]);
                let obs_m = evaluate(pm[0], pm[1], pm[2]);
                for r in 0..n_resid {
                    jac[(r, pi)] = (obs_p[r] - obs_m[r]) / (2.0 * eps * pdg_sigma[r]);
                }
            }
            let jt = jac.transpose();
            let jtj = &jt * &jac;
            let jtr = &jt * DVector::from_row_slice(&res);
            let lambda_ms = 0.1;
            let mut jtj_d = jtj.clone();
            for i in 0..n_params {
                jtj_d[(i, i)] += lambda_ms;
            }
            let delta = match jtj_d.lu().solve(&(-&jtr)) {
                Some(d) => d,
                None => break,
            };

            let mut step = 1.0;
            for _ in 0..10 {
                let np = [
                    (p[0] + step * delta[0]).max(0.01).min(2.0),
                    (p[1] + step * delta[1]).max(0.0).min(20.0),
                    (p[2] + step * delta[2]).max(0.0).min(10.0),
                ];
                let no = evaluate(np[0], np[1], np[2]);
                let nc: f64 = (0..n_resid)
                    .map(|i| ((no[i] - pdg_targets[i]) / pdg_sigma[i]).powi(2))
                    .sum();
                if nc < cost {
                    p = np;
                    break;
                }
                step *= 0.5;
            }
            if delta.norm() < 1e-6 {
                break;
            }
        }
        let obs = evaluate(p[0], p[1], p[2]);
        let cost: f64 = (0..n_resid)
            .map(|i| ((obs[i] - pdg_targets[i]) / pdg_sigma[i]).powi(2))
            .sum();
        if cost < global_best_cost {
            global_best_cost = cost;
            global_best_params = p;
        }
    }

    if global_best_cost < init_cost {
        params = global_best_params;
        println!(
            "\n  Multi-start found better minimum: cost = {:.2}",
            global_best_cost
        );
    }

    let final_obs = evaluate(params[0], params[1], params[2]);
    let final_cost: f64 = (0..n_resid)
        .map(|i| ((final_obs[i] - pdg_targets[i]) / pdg_sigma[i]).powi(2))
        .sum();

    println!("\n  === FINAL RESULT ===");
    println!(
        "  beta = {:.4}, alpha_ch = {:.4}, alpha_nu = {:.4}",
        params[0], params[1], params[2]
    );
    println!(
        "  theta_12 = {:.2} deg (PDG: {:.2}, err: {:.1}%)",
        final_obs[0],
        pdg.theta_12_deg,
        ((final_obs[0] - pdg.theta_12_deg) / pdg.theta_12_deg * 100.0).abs()
    );
    println!(
        "  theta_13 = {:.2} deg (PDG: {:.2}, err: {:.1}%)",
        final_obs[1],
        pdg.theta_13_deg,
        ((final_obs[1] - pdg.theta_13_deg) / pdg.theta_13_deg * 100.0).abs()
    );
    println!(
        "  theta_23 = {:.2} deg (PDG: {:.2}, err: {:.1}%)",
        final_obs[2],
        pdg.theta_23_deg,
        ((final_obs[2] - pdg.theta_23_deg) / pdg.theta_23_deg * 100.0).abs()
    );
    println!(
        "  r = {:.6} (PDG: {:.4}, err: {:.1}%)",
        final_obs[3],
        pdg_r,
        ((final_obs[3] - pdg_r) / pdg_r * 100.0).abs()
    );
    println!("  cost = {:.4} (4 residuals)", final_cost);
}

/// Friction-native baseline: M_ij = <profile_i, profile_j> (Gram matrix).
///
/// No additive Casimir baseline. The mass matrix IS the friction Gram
/// matrix directly. Diagonal = self-overlap (mass scale), off-diagonal =
/// cross-generation overlap (mixing).
///
/// Uses 3-blade profiles for the Gram construction, with psi coupling
/// for cross-generation terms. The alpha parameter scales the psi
/// contribution relative to the direct overlap.
#[test]
fn test_friction_native_baseline() {
    use crate::{
        bell_inequality::{SignTableCache, rotate_sparse},
        lepton_mass_hierarchy::cd_braid_signed_friction,
        majorana_braiding::MajoranaMode,
        three_fermion_generations::get_sedenion_subalgebras,
    };
    use cd_kernel::gourlay_psi;

    let pdg = Pdg2024::default();
    let pdg_r = 0.0307_f64;

    let (o1, o2, o3) = get_sedenion_subalgebras();
    let subs = [&o1, &o2, &o3];
    let sign_table = SignTableCache::new(16);

    // Use BOTH 3-blade and 2-blade profiles
    let build_pair_profile = |sel: (usize, usize), sub: &[usize]| -> [f64; 16] {
        let i = sel.0;
        let j = sel.1;
        let a_sparse = vec![(i, 1.0)];
        let a_rotated = rotate_sparse(&a_sparse, i, j, std::f64::consts::FRAC_PI_4);
        let b_sparse = vec![(j, 1.0)];
        let mut profile = [0.0_f64; 16];
        for &k in sub {
            if k == 0 || k == i || k == j {
                continue;
            }
            let x_sparse = [(k, 1.0)];
            profile[k] = sign_table.sparse_associator_sum(&a_rotated, &x_sparse, &b_sparse);
        }
        profile
    };

    // 2-blade profiles for angle-optimal mixing
    let ch_2b: Vec<[f64; 16]> = subs
        .iter()
        .map(|s| build_pair_profile((11, 12), s))
        .collect();
    let nu_2b: Vec<[f64; 16]> = subs.iter().map(|s| build_pair_profile((7, 8), s)).collect();

    let dot16 =
        |a: &[f64; 16], b: &[f64; 16]| -> f64 { a.iter().zip(b.iter()).map(|(x, y)| x * y).sum() };

    // 3-blade scalar frictions for mass hierarchy
    let sel_ch_3: Vec<f64> = subs
        .iter()
        .map(|s| {
            let ma = MajoranaMode {
                gamma_index: 0,
                cd_basis_index: 1,
                cd_dim: 16,
            };
            let mb = MajoranaMode {
                gamma_index: 5,
                cd_basis_index: 6,
                cd_dim: 16,
            };
            let mc = MajoranaMode {
                gamma_index: 10,
                cd_basis_index: 11,
                cd_dim: 16,
            };
            cd_braid_signed_friction(&ma, &mb, s, &sign_table)
                + cd_braid_signed_friction(&ma, &mc, s, &sign_table)
                + cd_braid_signed_friction(&mb, &mc, s, &sign_table)
        })
        .collect();
    let sel_nu_3: Vec<f64> = subs
        .iter()
        .map(|s| {
            let ma = MajoranaMode {
                gamma_index: 0,
                cd_basis_index: 1,
                cd_dim: 16,
            };
            let mb = MajoranaMode {
                gamma_index: 2,
                cd_basis_index: 3,
                cd_dim: 16,
            };
            let mc = MajoranaMode {
                gamma_index: 7,
                cd_basis_index: 8,
                cd_dim: 16,
            };
            cd_braid_signed_friction(&ma, &mb, s, &sign_table)
                + cd_braid_signed_friction(&ma, &mc, s, &sign_table)
                + cd_braid_signed_friction(&mb, &mc, s, &sign_table)
        })
        .collect();

    println!("  === Friction-Native Baseline (No Casimir) ===\n");
    println!(
        "  3-blade ch frictions: [{:.2}, {:.2}, {:.2}]",
        sel_ch_3[0], sel_ch_3[1], sel_ch_3[2]
    );
    println!(
        "  3-blade nu frictions: [{:.2}, {:.2}, {:.2}]",
        sel_nu_3[0], sel_nu_3[1], sel_nu_3[2]
    );

    // Scan: beta scales 3-blade diagonal, alpha scales 2-blade off-diagonal
    println!(
        "\n  {:>6} {:>6} | {:>8} {:>8} {:>8} | {:>8} | {:>8}",
        "beta", "alpha", "t12", "t13", "t23", "r", "cost"
    );

    let mut best = (f64::MAX, 0.0_f64, 0.0, [0.0_f64; 4]);

    for beta_step in 1..=30 {
        let beta = beta_step as f64 * 0.1;
        for alpha_step in 0..=30 {
            let alpha = alpha_step as f64 * 0.5;

            // Build mass matrix: M_ij = beta * exp(3blade) * delta_ij + alpha * <2blade_i, psi(2blade_j)>
            let mut m_nu = faer::Mat::zeros(3, 3);
            let mut m_ch = faer::Mat::zeros(3, 3);

            // Diagonal: exp(3-blade friction)
            let w1 = -0.656850_f64;
            let w2 = -0.741999_f64;
            for g in 0..3 {
                let f_ch = beta * (w1 * sel_ch_3[g] + w2 * sel_nu_3[g]);
                let f_nu = beta * (w1 * sel_nu_3[g] + w2 * sel_ch_3[g]);
                m_ch[(g, g)] = f_ch.exp();
                m_nu[(g, g)] = f_nu.exp();
            }

            // Off-diagonal: 2-blade psi coupling (NO Casimir baseline)
            for i in 0..3 {
                for j in 0..3 {
                    if i == j {
                        continue;
                    }
                    let psi_nu_j = gourlay_psi(&nu_2b[j]);
                    let psi_ch_j = gourlay_psi(&ch_2b[j]);
                    m_nu[(i, j)] = alpha * dot16(&nu_2b[i], &psi_nu_j);
                    m_ch[(i, j)] = alpha * dot16(&ch_2b[i], &psi_ch_j);
                }
            }

            let m_ch_s = (&m_ch + m_ch.transpose()) * faer::Scale(0.5);
            let m_nu_s = (&m_nu + m_nu.transpose()) * faer::Scale(0.5);
            let eig_ch = m_ch_s.self_adjoint_eigen(faer::Side::Lower).unwrap();
            let eig_nu = m_nu_s.self_adjoint_eigen(faer::Side::Lower).unwrap();
            let u_raw = eig_ch.U().transpose() * eig_nu.U();
            let (u_pmns, _, _) = crate::quark_sector::extract_ckm_permutation_aware(&u_raw);
            let (t12, t13, t23) = extract_pmns_angles(&u_pmns);

            let mut ev: Vec<f64> = (0..3)
                .map(|i| eig_nu.S().column_vector()[i].abs())
                .collect();
            ev.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let dm21 = ev[1] * ev[1] - ev[0] * ev[0];
            let dm31 = ev[2] * ev[2] - ev[0] * ev[0];
            let r = if dm31.abs() > 1e-30 { dm21 / dm31 } else { 1.0 };

            let cost = ((t12 - pdg.theta_12_deg) / pdg.theta_12_err).powi(2)
                + ((t13 - pdg.theta_13_deg) / pdg.theta_13_err).powi(2)
                + ((t23 - pdg.theta_23_deg) / pdg.theta_23_err).powi(2)
                + ((r - pdg_r) / 0.003).powi(2);

            if cost < best.0 {
                best = (cost, beta, alpha, [t12, t13, t23, r]);
            }
        }
    }

    let obs = best.3;
    println!("\n  === BEST FRICTION-NATIVE RESULT ===");
    println!("  beta = {:.1}, alpha = {:.1}", best.1, best.2);
    println!(
        "  theta_12 = {:.2} deg (PDG: {:.2}, err: {:.1}%)",
        obs[0],
        pdg.theta_12_deg,
        ((obs[0] - pdg.theta_12_deg) / pdg.theta_12_deg * 100.0).abs()
    );
    println!(
        "  theta_13 = {:.2} deg (PDG: {:.2}, err: {:.1}%)",
        obs[1],
        pdg.theta_13_deg,
        ((obs[1] - pdg.theta_13_deg) / pdg.theta_13_deg * 100.0).abs()
    );
    println!(
        "  theta_23 = {:.2} deg (PDG: {:.2}, err: {:.1}%)",
        obs[2],
        pdg.theta_23_deg,
        ((obs[2] - pdg.theta_23_deg) / pdg.theta_23_deg * 100.0).abs()
    );
    println!(
        "  r = {:.4} (PDG: {:.4}, err: {:.1}%)",
        obs[3],
        pdg_r,
        ((obs[3] - pdg_r) / pdg_r * 100.0).abs()
    );
    println!("  cost = {:.2}", best.0);
}

/// Weinberg angle from G2 stabilizer decomposition.
///
/// Multiple approaches:
/// 1. GUT: sin^2(theta_W) = 3/8 = 0.375 (SU(5) at unification)
/// 2. Flux ratio: SU(2)/SU(3) associator flux = 0.529 -> 0.199 (C-1458)
/// 3. Dimensional: various dim ratios from G2 decomposition
/// 4. Casimir: C2(SU(2))/C2(SU(3)) ratios
/// 5. NEW: G2 stabilizer norm decomposition
#[test]
fn test_weinberg_angle_from_g2() {
    use gororoba_algebra::lie::g2_stabilizer::{stabilizer_decomposition, structure_constants};

    let pdg_sin2_tw = 0.2312_f64;

    println!("  === Weinberg Angle from G2 Stabilizer ===\n");
    println!("  PDG: sin^2(theta_W) = {:.4}\n", pdg_sin2_tw);

    // Approach 1: GUT SU(5)
    let gut = 3.0 / 8.0;
    println!(
        "  1. SU(5) GUT: sin^2 = 3/8 = {:.4} (err: {:.1}%)",
        gut,
        ((gut - pdg_sin2_tw) / pdg_sin2_tw * 100.0).abs()
    );

    // Approach 2: Existing flux ratio
    let flux = 0.199;
    println!(
        "  2. Flux ratio (C-1458): sin^2 = {:.4} (err: {:.1}%)",
        flux,
        ((flux - pdg_sin2_tw) / pdg_sin2_tw * 100.0).abs()
    );

    // Approach 3: G2 dim decomposition
    let dim_stab = 8.0_f64;
    let dim_coset = 6.0_f64;
    let dim_g2 = 14.0_f64;

    // sin^2 = dim(coset) / dim(G2) * correction?
    let dim_ratio_a = dim_coset / dim_g2; // 6/14 = 0.429
    let _dim_ratio_b = dim_coset / (dim_stab + dim_coset); // = 6/14 same
    let dim_ratio_c = 1.0 / (1.0 + dim_stab / 3.0); // 1/(1+8/3) = 3/11 = 0.273

    println!(
        "  3a. dim(coset)/dim(G2) = 6/14 = {:.4} (err: {:.1}%)",
        dim_ratio_a,
        ((dim_ratio_a - pdg_sin2_tw) / pdg_sin2_tw * 100.0).abs()
    );
    println!(
        "  3b. 3/(3+8) = 3/11 = {:.4} (err: {:.1}%)",
        dim_ratio_c,
        ((dim_ratio_c - pdg_sin2_tw) / pdg_sin2_tw * 100.0).abs()
    );

    // Approach 4: Casimir ratios
    // C2(fund, SU(3)) = 4/3, C2(fund, SU(2)) = 3/4
    let c2_su3 = 4.0 / 3.0;
    let c2_su2 = 3.0 / 4.0;
    let casimir_ratio = c2_su2 / (c2_su2 + c2_su3); // 0.75/(0.75+1.33) = 0.36
    println!(
        "  4. C2(SU2)/(C2(SU2)+C2(SU3)) = {:.4} (err: {:.1}%)",
        casimir_ratio,
        ((casimir_ratio - pdg_sin2_tw) / pdg_sin2_tw * 100.0).abs()
    );

    // Approach 5: G2 stabilizer structure constants
    // The total contraction sum f_{abc}^2 for the stabilizer gives the
    // dual Coxeter number h* times dim. For SU(3): h*=3, dim=8, total=24.
    // For the coset: compute the coset structure constants and their total.
    let decomp = stabilizer_decomposition(1);
    let f_stab = structure_constants(&decomp.stabilizer_basis);

    let mut total_stab = 0.0_f64;
    for a in 0..8 {
        for b in 0..8 {
            for c in 0..8 {
                total_stab += f_stab[a][b][c] * f_stab[a][b][c];
            }
        }
    }

    let f_coset = structure_constants(&decomp.coset_complement);
    let mut total_coset = 0.0_f64;
    for a in 0..6 {
        for b in 0..6 {
            for c in 0..6 {
                total_coset += f_coset[a][b][c] * f_coset[a][b][c];
            }
        }
    }

    println!("\n  5. G2 structure constant decomposition:");
    println!("     sum f_stab^2 = {:.2} (expected: 24 = 3*8)", total_stab);
    println!("     sum f_coset^2 = {:.2}", total_coset);

    let sc_ratio = total_coset / (total_stab + total_coset);
    println!(
        "     f_coset^2 / (f_stab^2 + f_coset^2) = {:.4} (err: {:.1}%)",
        sc_ratio,
        ((sc_ratio - pdg_sin2_tw) / pdg_sin2_tw * 100.0).abs()
    );

    // Approach 6: Use the Fano line structure
    // Each e_k lies on 3 Fano lines (21 edges total for 7 points).
    // The Weinberg angle relates to the SU(2)/SU(3) coupling ratio.
    // In the Fano plane: 7 points, 7 lines, 3 points per line.
    // The ratio 3/7 = 0.429 and 1/7 = 0.143...
    // Try: (7 - 3*2) / (7 + 7) = 1/14 = 0.071... nope.
    //
    // Actually: sin^2(theta_W) = g'^2/(g^2+g'^2).
    // At the GUT scale, g = g' = g_5, so sin^2 = 1/(1+1) * normalization.
    // The normalization is 3/5 for SU(5): sin^2 = 3/(3+5) = 3/8.
    //
    // In the G2 framework: the SU(3) stabilizer has 8 generators,
    // the coset has 6 directions. If we identify the U(1) generator
    // as a specific coset direction, then:
    // g'^2 proportional to 1 (one U(1) generator)
    // g^2 proportional to 3 (three SU(2) generators)
    // sin^2 = 1/(1+3) = 1/4 = 0.25
    let simple = 0.25;
    println!(
        "\n  6. 1/(1+3) = 1/4 = {:.4} (err: {:.1}%)",
        simple,
        ((simple - pdg_sin2_tw) / pdg_sin2_tw * 100.0).abs()
    );

    // Best match summary
    println!("\n  === SUMMARY ===");
    let approaches = [
        ("SU(5) GUT 3/8", gut),
        ("Flux ratio", flux),
        ("dim 6/14", dim_ratio_a),
        ("dim 3/11", dim_ratio_c),
        ("Casimir ratio", casimir_ratio),
        ("f^2 coset/total", sc_ratio),
        ("1/4 simple", simple),
    ];
    for (name, val) in &approaches {
        let err = ((val - pdg_sin2_tw) / pdg_sin2_tw * 100.0).abs();
        let marker = if err < 10.0 { " <-- BEST" } else { "" };
        println!("    {:>20}: {:.4} (err: {:.1}%){}", name, val, err, marker);
    }
}

/// Weinberg angle: 1-loop RG running from G2 tree-level to M_Z.
///
/// sin^2(theta_W, mu) = sin^2(M_GUT) + (b_1 - b_2) / (2*pi) * alpha * ln(mu/M_GUT)
/// where b_1 = 41/10, b_2 = -19/6 are the SM 1-loop beta coefficients.
#[test]
fn test_weinberg_running() {
    let sin2_gut = 0.250_f64; // G2 tree-level prediction (= 1/4)
    let m_z = 91.1876_f64; // GeV
    let alpha_em = 1.0 / 127.9_f64; // at M_Z

    // SM 1-loop beta coefficients (U(1)_Y and SU(2)_L)
    let b1 = 41.0 / 10.0_f64; // U(1)_Y
    let b2 = -19.0 / 6.0_f64; // SU(2)_L

    println!("  === Weinberg Angle RG Running ===\n");
    println!("  Tree-level (G2): sin^2 = {:.4}", sin2_gut);
    println!("  PDG at M_Z: sin^2 = 0.2312\n");

    // Scan unification scales
    for log_m_gut in [14.0, 15.0, 15.5, 16.0, 16.5, 17.0, 18.0_f64] {
        let m_gut = 10.0_f64.powf(log_m_gut);
        // 1-loop running: Delta(sin^2) = (3/5) * alpha/(2*pi) * (b1 - b2) * ln(M_Z/M_GUT)
        // The 3/5 is the SU(5) normalization factor for U(1)_Y
        let delta =
            (3.0 / 5.0) * alpha_em / (2.0 * std::f64::consts::PI) * (b1 - b2) * (m_z / m_gut).ln();
        let sin2_mz = sin2_gut + delta;
        let err = ((sin2_mz - 0.2312) / 0.2312 * 100.0).abs();
        let marker = if err < 3.0 { " <--" } else { "" };
        println!(
            "  M_GUT = 10^{:.1} GeV: sin^2(M_Z) = {:.4} (err: {:.1}%){}",
            log_m_gut, sin2_mz, err, marker
        );
    }

    // The G2-specific unification scale: related to the G2 manifold volume
    // In string/M-theory, G2 holonomy manifolds have characteristic scale
    // around 10^16 GeV. Let's check.
    let m_gut_g2 = 1e16_f64;
    let delta_g2 =
        (3.0 / 5.0) * alpha_em / (2.0 * std::f64::consts::PI) * (b1 - b2) * (m_z / m_gut_g2).ln();
    let sin2_mz_g2 = sin2_gut + delta_g2;
    println!("\n  At M_G2 = 10^16 GeV (G2 holonomy scale):");
    println!(
        "  sin^2(M_Z) = {:.4} (PDG: 0.2312, err: {:.1}%)",
        sin2_mz_g2,
        ((sin2_mz_g2 - 0.2312) / 0.2312 * 100.0).abs()
    );
}

/// Full 3-blade model: 3-blade diagonal + 3-blade off-diagonal.
///
/// Off-diagonal: sum of 3 pairwise psi overlaps (3x amplitude of 2-blade).
/// This should provide enough off-diagonal strength to rotate eigenvectors
/// while preserving the 3-blade mass hierarchy.
#[test]
fn test_full_3blade_model() {
    use crate::{
        bell_inequality::{SignTableCache, rotate_sparse},
        lepton_mass_hierarchy::cd_braid_signed_friction,
        majorana_braiding::MajoranaMode,
        three_fermion_generations::get_sedenion_subalgebras,
    };
    use cd_kernel::gourlay_psi;

    let pdg = Pdg2024::default();
    let pdg_r = 0.0307_f64;

    let (o1, o2, o3) = get_sedenion_subalgebras();
    let subs = [&o1, &o2, &o3];
    let sign_table = SignTableCache::new(16);
    let w1 = -0.656850_f64;
    let w2 = -0.741999_f64;

    // Build 3-blade profiles for BOTH diagonal and off-diagonal
    let build_3blade_profile = |a: usize, b: usize, c: usize, sub: &[usize]| -> [f64; 16] {
        let ma = MajoranaMode {
            gamma_index: a - 1,
            cd_basis_index: a,
            cd_dim: 16,
        };
        let mb = MajoranaMode {
            gamma_index: b - 1,
            cd_basis_index: b,
            cd_dim: 16,
        };
        let mc = MajoranaMode {
            gamma_index: c - 1,
            cd_basis_index: c,
            cd_dim: 16,
        };
        let build_pair = |m1: &MajoranaMode, m2: &MajoranaMode, s: &[usize]| -> [f64; 16] {
            let i = m1.cd_basis_index;
            let j = m2.cd_basis_index;
            let a_sp = vec![(i, 1.0)];
            let a_rot = rotate_sparse(&a_sp, i, j, std::f64::consts::FRAC_PI_4);
            let b_sp = vec![(j, 1.0)];
            let mut p = [0.0_f64; 16];
            for &k in s {
                if k == 0 || k == i || k == j {
                    continue;
                }
                p[k] = sign_table.sparse_associator_sum(&a_rot, &[(k, 1.0)], &b_sp);
            }
            p
        };
        let p_ab = build_pair(&ma, &mb, sub);
        let p_ac = build_pair(&ma, &mc, sub);
        let p_bc = build_pair(&mb, &mc, sub);
        let mut combined = [0.0_f64; 16];
        for idx in 0..16 {
            combined[idx] = p_ab[idx] + p_ac[idx] + p_bc[idx];
        }
        combined
    };

    // Use angle-optimal triple for off-diagonal (different from mass-ratio triple)
    // ch: (11,12) pair embedded as triple (10,11,12) -- includes neighbors
    // nu: (7,8) pair embedded as triple (7,8,9) -- includes neighbors
    let ch_profiles: Vec<[f64; 16]> = subs
        .iter()
        .map(|s| build_3blade_profile(10, 11, 12, s))
        .collect();
    let nu_profiles: Vec<[f64; 16]> = subs
        .iter()
        .map(|s| build_3blade_profile(7, 8, 9, s))
        .collect();

    // Mass-ratio triple frictions (diagonal)
    let sel_ch_3: Vec<f64> = subs
        .iter()
        .map(|s| {
            let ma = MajoranaMode {
                gamma_index: 0,
                cd_basis_index: 1,
                cd_dim: 16,
            };
            let mb = MajoranaMode {
                gamma_index: 5,
                cd_basis_index: 6,
                cd_dim: 16,
            };
            let mc = MajoranaMode {
                gamma_index: 10,
                cd_basis_index: 11,
                cd_dim: 16,
            };
            cd_braid_signed_friction(&ma, &mb, s, &sign_table)
                + cd_braid_signed_friction(&ma, &mc, s, &sign_table)
                + cd_braid_signed_friction(&mb, &mc, s, &sign_table)
        })
        .collect();
    let sel_nu_3: Vec<f64> = subs
        .iter()
        .map(|s| {
            let ma = MajoranaMode {
                gamma_index: 0,
                cd_basis_index: 1,
                cd_dim: 16,
            };
            let mb = MajoranaMode {
                gamma_index: 2,
                cd_basis_index: 3,
                cd_dim: 16,
            };
            let mc = MajoranaMode {
                gamma_index: 7,
                cd_basis_index: 8,
                cd_dim: 16,
            };
            cd_braid_signed_friction(&ma, &mb, s, &sign_table)
                + cd_braid_signed_friction(&ma, &mc, s, &sign_table)
                + cd_braid_signed_friction(&mb, &mc, s, &sign_table)
        })
        .collect();

    let dot16 =
        |a: &[f64; 16], b: &[f64; 16]| -> f64 { a.iter().zip(b.iter()).map(|(x, y)| x * y).sum() };

    println!("  === Full 3-Blade Model (3-blade diagonal + 3-blade off-diagonal) ===\n");

    // Check 3-blade off-diagonal amplitude vs 2-blade
    let psi_nu_0 = gourlay_psi(&nu_profiles[0]);
    let offdiag_3b = dot16(&nu_profiles[1], &psi_nu_0);
    println!("  3-blade off-diagonal amplitude: {:.2}", offdiag_3b);

    let mut best = (f64::MAX, 0.0_f64, 0.0, [0.0_f64; 4]);

    for beta_step in 1..=30 {
        let beta = beta_step as f64 * 0.1;
        for alpha_step in 0..=40 {
            let alpha = alpha_step as f64 * 0.5;

            let mut m_ch = faer::Mat::zeros(3, 3);
            let mut m_nu = faer::Mat::zeros(3, 3);

            // Diagonal: 3-blade (mass-ratio optimal)
            for g in 0..3 {
                let f_ch = beta * (w1 * sel_ch_3[g] + w2 * sel_nu_3[g]);
                let f_nu = beta * (w1 * sel_nu_3[g] + w2 * sel_ch_3[g]);
                m_ch[(g, g)] = f_ch.exp();
                m_nu[(g, g)] = f_nu.exp();
            }

            // Off-diagonal: 3-blade psi coupling (3x amplitude)
            for i in 0..3 {
                for j in 0..3 {
                    if i == j {
                        continue;
                    }
                    let psi_nu_j = gourlay_psi(&nu_profiles[j]);
                    let psi_ch_j = gourlay_psi(&ch_profiles[j]);
                    m_nu[(i, j)] = alpha * dot16(&nu_profiles[i], &psi_nu_j);
                    m_ch[(i, j)] = alpha * dot16(&ch_profiles[i], &psi_ch_j);
                }
            }

            let m_ch_s = (&m_ch + m_ch.transpose()) * faer::Scale(0.5);
            let m_nu_s = (&m_nu + m_nu.transpose()) * faer::Scale(0.5);
            let eig_ch = m_ch_s.self_adjoint_eigen(faer::Side::Lower).unwrap();
            let eig_nu = m_nu_s.self_adjoint_eigen(faer::Side::Lower).unwrap();
            let u_raw = eig_ch.U().transpose() * eig_nu.U();
            let (u_pmns, _, _) = crate::quark_sector::extract_ckm_permutation_aware(&u_raw);
            let (t12, t13, t23) = extract_pmns_angles(&u_pmns);

            let mut ev: Vec<f64> = (0..3)
                .map(|i| eig_nu.S().column_vector()[i].abs())
                .collect();
            ev.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let dm21 = ev[1] * ev[1] - ev[0] * ev[0];
            let dm31 = ev[2] * ev[2] - ev[0] * ev[0];
            let r = if dm31.abs() > 1e-30 { dm21 / dm31 } else { 1.0 };

            let cost = ((t12 - pdg.theta_12_deg) / pdg.theta_12_err).powi(2)
                + ((t13 - pdg.theta_13_deg) / pdg.theta_13_err).powi(2)
                + ((t23 - pdg.theta_23_deg) / pdg.theta_23_err).powi(2)
                + ((r - pdg_r) / 0.003).powi(2);

            if cost < best.0 {
                best = (cost, beta, alpha, [t12, t13, t23, r]);
            }
        }
    }

    let obs = best.3;
    println!("\n  === BEST FULL 3-BLADE RESULT ===");
    println!("  beta = {:.1}, alpha = {:.1}", best.1, best.2);
    println!(
        "  theta_12 = {:.2} deg (PDG: {:.2}, err: {:.1}%)",
        obs[0],
        pdg.theta_12_deg,
        ((obs[0] - pdg.theta_12_deg) / pdg.theta_12_deg * 100.0).abs()
    );
    println!(
        "  theta_13 = {:.2} deg (PDG: {:.2}, err: {:.1}%)",
        obs[1],
        pdg.theta_13_deg,
        ((obs[1] - pdg.theta_13_deg) / pdg.theta_13_deg * 100.0).abs()
    );
    println!(
        "  theta_23 = {:.2} deg (PDG: {:.2}, err: {:.1}%)",
        obs[2],
        pdg.theta_23_deg,
        ((obs[2] - pdg.theta_23_deg) / pdg.theta_23_deg * 100.0).abs()
    );
    println!(
        "  r = {:.4} (PDG: {:.4}, err: {:.1}%)",
        obs[3],
        pdg_r,
        ((obs[3] - pdg_r) / pdg_r * 100.0).abs()
    );
    println!("  cost = {:.2}", best.0);
}

/// Numerical regression baselines for the optimization refactor.
///
/// Small, deterministic checks that run in < 1s. Captures the known-good
/// outputs from commit 83c4254f so any performance optimization that
/// changes numerical results is caught immediately.
#[test]
fn test_numerical_regression_baselines() {
    use cd_kernel::cayley_dickson::SignTable;

    // --- Cardano eigensolver: known eigenvalues ---
    let h: [[C2; 3]; 3] = [
        [(3.0, 0.0), (0.5, 1.2), (-0.3, 0.8)],
        [(0.5, -1.2), (1.0, 0.0), (0.7, -0.4)],
        [(-0.3, -0.8), (0.7, 0.4), (2.0, 0.0)],
    ];
    let (evals, _) = hermitian_3x3_eig(&h);
    assert!((evals[0] - 0.061606).abs() < 1e-4, "eval[0]={}", evals[0]);
    assert!((evals[1] - 1.850281).abs() < 1e-4, "eval[1]={}", evals[1]);
    assert!((evals[2] - 4.088113).abs() < 1e-4, "eval[2]={}", evals[2]);

    // --- V_k basis dim=16: rank and leading singular value ---
    let (basis_16, sv_16, assess_16) = extract_vk_basis(16, 12);
    assert_eq!(assess_16.len(), 42, "sedenion assessor count");
    assert_eq!(basis_16.nrows(), 6, "sedenion V_k rank must be 6");
    assert!(
        (sv_16[0] - 3.419971).abs() < 1e-4,
        "sv[0]={:.6}, expected 3.419971",
        sv_16[0]
    );
    // All 6 SVs are degenerate
    for i in 1..6 {
        assert!(
            (sv_16[i] - sv_16[0]).abs() < 1e-4,
            "sv[{i}]={:.6} differs from sv[0]={:.6}",
            sv_16[i],
            sv_16[0]
        );
    }

    // Orthonormality: B * B^T = I_rank (basis vectors are ROWS)
    let bbt = &basis_16 * basis_16.transpose();
    let eye6 = nalgebra::DMatrix::identity(6, 6);
    let ortho_err = (&bbt - &eye6).norm();
    assert!(
        ortho_err < 1e-10,
        "basis non-orthonormal: |B*B^T - I|_F = {:.3e}",
        ortho_err
    );

    // --- SignTable dim=16: spot checks (exact integer) ---
    let stab = SignTable::new(16);
    assert_eq!(stab.sign(0, 0), 1, "sign(0,0) must be +1 (scalar*scalar)");
    for k in 0..16 {
        assert_eq!(
            stab.sign(0, k),
            1,
            "sign(0,{k}) must be +1 (e_0 is identity)"
        );
    }
    // Antisymmetry for imaginary units
    for p in 1..16 {
        for q in (p + 1)..16 {
            assert_eq!(
                stab.sign(p, q),
                -stab.sign(q, p),
                "antisymmetry violated at ({p},{q})"
            );
        }
    }
    // Self-products: e_p * e_p = -1 for p > 0
    for p in 1..16 {
        assert_eq!(stab.sign(p, p), -1, "sign({p},{p}) must be -1");
    }

    println!("  Numerical regression baselines: all checks passed");
}

/// Validate U(1) phase canonicalization in the Cardano eigensolver.
///
/// Constructs a known 3x3 Hermitian matrix with complex off-diagonal
/// entries, solves with both `hermitian_3x3_eig` (Cardano) and faer
/// (iterative QR), then verifies:
///
/// 1. Eigenvalue agreement: |lambda_cardano - lambda_faer| < 1e-12
/// 2. Projector agreement: |v*v^dag - v_faer*v_faer^dag|_F < 1e-10
/// 3. Residual: |Hv - lambda*v| / (|H|*|v|) < 1e-12
/// 4. Phase convention: largest-magnitude component is real and nonneg
#[test]
fn test_cardano_phase_canonicalization() {
    // Build a 3x3 Hermitian matrix with nontrivial complex structure
    let h: [[C2; 3]; 3] = [
        [(3.0, 0.0), (0.5, 1.2), (-0.3, 0.8)],
        [(0.5, -1.2), (1.0, 0.0), (0.7, -0.4)],
        [(-0.3, -0.8), (0.7, 0.4), (2.0, 0.0)],
    ];

    let (evals, evecs) = hermitian_3x3_eig(&h);

    // --- Check 4: phase convention ---
    for col in 0..3 {
        let mut max_mag_sq = 0.0_f64;
        let mut max_idx = 0;
        for i in 0..3 {
            let ms = evecs[i][col].0 * evecs[i][col].0 + evecs[i][col].1 * evecs[i][col].1;
            if ms > max_mag_sq {
                max_mag_sq = ms;
                max_idx = i;
            }
        }
        let (re, im) = evecs[max_idx][col];
        assert!(
            re >= -1e-14,
            "col {col}: largest component has re={re:.6e} (should be >= 0)"
        );
        assert!(
            im.abs() < 1e-12,
            "col {col}: largest component has im={im:.6e} (should be ~0)"
        );
    }

    // --- Check 3: residual |Hv - lam*v| ---
    let h_frob = {
        let mut s = 0.0_f64;
        for row in &h {
            for &(r, m) in row {
                s += r * r + m * m;
            }
        }
        s.sqrt()
    };
    for col in 0..3 {
        let lam = evals[col];
        let mut res_sq = 0.0_f64;
        for i in 0..3 {
            let mut hv = (0.0_f64, 0.0_f64);
            for j in 0..3 {
                let p = cmul(h[i][j], evecs[j][col]);
                hv.0 += p.0;
                hv.1 += p.1;
            }
            let diff_re = hv.0 - lam * evecs[i][col].0;
            let diff_im = hv.1 - lam * evecs[i][col].1;
            res_sq += diff_re * diff_re + diff_im * diff_im;
        }
        let residual = res_sq.sqrt() / (h_frob + lam.abs());
        assert!(
            residual < 1e-12,
            "col {col}: relative residual {residual:.3e} exceeds 1e-12"
        );
    }

    // --- Check 1: eigenvalue agreement with faer ---
    let mut h_faer = faer::Mat::<faer::c64>::zeros(3, 3);
    for i in 0..3 {
        for j in 0..3 {
            h_faer[(i, j)] = faer::c64::new(h[i][j].0, h[i][j].1);
        }
    }
    let eig_faer = h_faer.self_adjoint_eigen(faer::Side::Lower).unwrap();
    let mut faer_evals = [0.0_f64; 3];
    for i in 0..3 {
        faer_evals[i] = eig_faer.S().column_vector()[i].re;
    }
    faer_evals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    for i in 0..3 {
        let diff = (evals[i] - faer_evals[i]).abs();
        assert!(
            diff < 1e-12,
            "eigenvalue {i}: cardano={:.12e}, faer={:.12e}, diff={diff:.3e}",
            evals[i],
            faer_evals[i]
        );
    }

    // --- Check 2: projector agreement |v*v^dag - v_faer*v_faer^dag|_F ---
    // Match Cardano eigenvectors to faer by closest eigenvalue
    for col in 0..3 {
        // Find faer column with closest eigenvalue
        let faer_col = (0..3)
            .min_by(|&a, &b| {
                let da = (evals[col] - eig_faer.S().column_vector()[a].re).abs();
                let db = (evals[col] - eig_faer.S().column_vector()[b].re).abs();
                da.partial_cmp(&db).unwrap()
            })
            .unwrap();

        // Compute |P_cardano - P_faer|_F^2
        let mut frob_sq = 0.0_f64;
        for i in 0..3 {
            for j in 0..3 {
                // P_cardano[i][j] = v_i * conj(v_j)
                let pc = cmul(evecs[i][col], cconj(evecs[j][col]));
                // P_faer[i][j] = u_i * conj(u_j)
                let ui = eig_faer.U()[(i, faer_col)];
                let uj = eig_faer.U()[(j, faer_col)];
                let pf_re = ui.re * uj.re + ui.im * uj.im;
                let pf_im = ui.im * uj.re - ui.re * uj.im;
                let dr = pc.0 - pf_re;
                let di = pc.1 - pf_im;
                frob_sq += dr * dr + di * di;
            }
        }
        let frob = frob_sq.sqrt();
        assert!(
            frob < 1e-10,
            "col {col}: projector Frobenius distance {frob:.3e} exceeds 1e-10"
        );
    }

    println!("  Cardano phase canonicalization: all 4 checks passed");
    println!(
        "  Eigenvalues: [{:.6}, {:.6}, {:.6}]",
        evals[0], evals[1], evals[2]
    );
}

/// Validate rephasing-invariant delta_CP extraction against known PDG
/// parametrization.
///
/// Constructs a PMNS matrix from known angles and delta using the PDG
/// standard parametrization, extracts moduli and Jarlskog, then
/// verifies that `extract_delta_cp_invariant` recovers the input delta.
#[test]
fn test_delta_cp_invariant_extraction() {
    // PDG best-fit values (NuFIT 5.3, NO)
    let t12 = 33.41_f64.to_radians();
    let t13 = 8.54_f64.to_radians();
    let t23 = 49.0_f64.to_radians();

    // Test several delta values including the PDG best-fit
    let test_deltas = [195.0_f64, 93.0, 270.0, 45.0, 0.0, 180.0, -30.0];

    let (s12, c12) = (t12.sin(), t12.cos());
    let (s13, c13) = (t13.sin(), t13.cos());
    let (s23, c23) = (t23.sin(), t23.cos());

    for &delta_deg in &test_deltas {
        let delta = delta_deg.to_radians();
        let (sd, cd) = (delta.sin(), delta.cos());

        // Build PMNS moduli from PDG parametrization
        // U_e1 = c12*c13, U_e2 = s12*c13, U_e3 = s13
        // U_mu1 = -s12*c23 - c12*s23*s13*exp(i*delta)
        // |U_mu1|^2 = s12^2*c23^2 + c12^2*s23^2*s13^2
        //             + 2*s12*c12*s23*c23*s13*cos(delta)
        let u_mu1_sq = s12 * s12 * c23 * c23
            + c12 * c12 * s23 * s23 * s13 * s13
            + 2.0 * s12 * c12 * s23 * c23 * s13 * cd;

        // U_mu2 = c12*c23 - s12*s23*s13*exp(i*delta)
        let u_mu2_sq = c12 * c12 * c23 * c23 + s12 * s12 * s23 * s23 * s13 * s13
            - 2.0 * c12 * s12 * s23 * c23 * s13 * cd;

        // U_tau1 = s12*s23 - c12*c23*s13*exp(i*delta)
        let u_tau1_sq = s12 * s12 * s23 * s23 + c12 * c12 * c23 * c23 * s13 * s13
            - 2.0 * s12 * c12 * c23 * s23 * s13 * cd;

        // U_tau2 = -c12*s23 - s12*c23*s13*exp(i*delta)
        let u_tau2_sq = c12 * c12 * s23 * s23
            + s12 * s12 * c23 * c23 * s13 * s13
            + 2.0 * c12 * s12 * c23 * s23 * s13 * cd;

        let u_moduli = [
            [c12 * c13, s12 * c13, s13],
            [u_mu1_sq.sqrt(), u_mu2_sq.sqrt(), s23 * c13],
            [u_tau1_sq.sqrt(), u_tau2_sq.sqrt(), c23 * c13],
        ];

        // Jarlskog = s12*c12*s23*c23*s13*c13^2*sin(delta)
        let j_cp = s12 * c12 * s23 * c23 * s13 * c13 * c13 * sd;

        let recovered = extract_delta_cp_invariant(&u_moduli, j_cp);

        // Wrap both to [-180, 180] for comparison
        let wrap = |x: f64| -> f64 {
            let mut v = x % 360.0;
            if v > 180.0 {
                v -= 360.0;
            }
            if v < -180.0 {
                v += 360.0;
            }
            v
        };
        let diff = (wrap(recovered) - wrap(delta_deg)).abs();
        let diff = if diff > 180.0 { 360.0 - diff } else { diff };

        println!(
            "  delta_in={:7.1} deg  recovered={:7.1} deg  diff={:.2e}",
            delta_deg, recovered, diff
        );

        // Skip delta=0 and delta=180 where cos(delta) is degenerate
        if delta_deg.abs() > 1.0 && (delta_deg - 180.0).abs() > 1.0 {
            assert!(
                diff < 0.1,
                "delta={delta_deg}: recovered={recovered:.2}, diff={diff:.2e}"
            );
        }
    }
}

/// Phase-only CP violation via J_k complex structure (C-1494).
///
/// # Physical mechanism
///
/// Multiply off-diagonal mass-matrix elements by a phase factor:
///
/// ```text
/// M[i][j] -> |M[i][j]| * exp(i * alpha_CP * phi[i][j])
/// ```
///
/// This preserves eigenvalue magnitudes (diagonal stays real,
/// Hermiticity imposed via `M[j][i] = conj(M[i][j])`) while
/// introducing rephasing-invariant CP violation through a nonzero
/// Jarlskog invariant J = Im(U_e1 * U_mu2 * conj(U_e2) * conj(U_mu1)).
///
/// # Phase angle derivation
///
/// The phase `phi[i][j]` is the natural Fano-derived complex angle:
///
/// ```text
/// phi[i][j] = atan2(<profile_i, J_k(psi(profile_j))>,
///                    <profile_i, psi(profile_j)>)
/// ```
///
/// where `J_k` is the full 16D complex structure from
/// [`apply_jk_full_16d`] acting on both octonion halves, and `psi`
/// is the Gourlay zero-divisor map from `cd_kernel::gourlay_psi`.
///
/// # Why multiplicative phase, not additive imaginary
///
/// The earlier additive approach `M += i * alpha * template` distorted
/// mixing angles by 50-300% because it changed eigenvalue magnitudes.
/// The multiplicative approach preserves `|M[i][j]|` exactly, so
/// angle distortion stays below 1.5% across the entire alpha_CP scan.
///
/// # Why full 16D J_k (C-1496)
///
/// Originally used the 6D perp-only `ComplexStructure::matrix` action.
/// Upgraded to [`apply_jk_full_16d`] to test whether the upper
/// octonion block contributes additional phase.  Result: NULL -- both
/// actions produce identical |J_CP| because the friction profiles
/// from selectors (e_7, e_8) have zero upper-block components.
///
/// # Expected output
///
/// For each k=1..7, prints: alpha_CP, mixing angles with % error,
/// |J_CP|, delta_CP.  Best result: |J_CP| ~ 8.5e-3 (25% of PDG).
///
/// # Claims exercised
///
/// - C-1494: Phase-only CP violation baseline
/// - C-1496: 16D vs 6D null result (this test now uses 16D)
#[test]
fn test_cp_violation_phase_only() {
    use crate::{
        bell_inequality::{SignTableCache, rotate_sparse},
        majorana_braiding::MajoranaMode,
        three_fermion_generations::get_sedenion_subalgebras,
    };
    use cd_kernel::gourlay_psi;

    let ch_pair = (11_usize, 12);
    let nu_pair = (7_usize, 8);
    let alpha_ch = 3.00_f64;
    let alpha_nu = 1.35_f64;

    // Get the best real mass matrices + V_6 correction
    let (m_ch_real, m_nu_real) =
        construct_pmns_matrices_two_param(ch_pair, nu_pair, alpha_ch, alpha_nu);
    let (v6_basis, _, _) = extract_v6_basis();
    let lift = TensorElementLift;
    let eps = 0.05_f64;
    let n_basis = v6_basis.nrows().min(6);

    // Compute constrained directions
    let eig_ch_0 = m_ch_real.self_adjoint_eigen(faer::Side::Lower).unwrap();
    let eig_nu_0 = m_nu_real.self_adjoint_eigen(faer::Side::Lower).unwrap();
    let u_raw_0 = eig_ch_0.U().transpose() * eig_nu_0.U();
    let (_, perm_u, perm_d) = crate::quark_sector::extract_ckm_permutation_aware(&u_raw_0);

    let angles_at = |beta: &[f64; 6]| -> (f64, f64, f64) {
        let mut m_nu = m_nu_real.clone();
        apply_v6_perturbation(&mut m_nu, &v6_basis, beta, &lift);
        let m_nu_s = (&m_nu + m_nu.transpose()) * faer::Scale(0.5);
        let eig_nu = m_nu_s.self_adjoint_eigen(faer::Side::Lower).unwrap();
        let u_raw = eig_ch_0.U().transpose() * eig_nu.U();
        let mut u_perm = faer::Mat::zeros(3, 3);
        for i in 0..3 {
            for j in 0..3 {
                u_perm[(i, j)] = u_raw[(perm_u[i], perm_d[j])];
            }
        }
        extract_pmns_angles(&u_perm)
    };

    // Compute gradients via finite differences
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

    let inner_angles = |t_sol: f64, t_atm: f64| -> (f64, f64, f64) {
        let mut beta = [0.0_f64; 6];
        for k in 0..6 {
            beta[k] = t_sol * u_solar[k] + t_atm * u_atmo[k];
        }
        angles_at(&beta)
    };
    let (t_sol, t_atm, _, _) = gauss_newton_2d(
        &inner_angles,
        1.5,
        0.0,
        (33.41, 8.54, 49.0),
        (1.0, 2.24, 1.0),
        15,
    );

    let mut beta_opt = [0.0_f64; 6];
    for k in 0..6 {
        beta_opt[k] = t_sol * u_solar[k] + t_atm * u_atmo[k];
    }
    let mut m_nu_corrected = m_nu_real.clone();
    apply_v6_perturbation(&mut m_nu_corrected, &v6_basis, &beta_opt, &lift);
    let m_nu_corrected = (&m_nu_corrected + m_nu_corrected.transpose()) * faer::Scale(0.5);

    // Verify baseline angles
    let eig_nu_c0 = m_nu_corrected
        .self_adjoint_eigen(faer::Side::Lower)
        .unwrap();
    let u_real_baseline = eig_ch_0.U().transpose() * eig_nu_c0.U();
    let mut u_perm_base = faer::Mat::zeros(3, 3);
    for i in 0..3 {
        for j in 0..3 {
            u_perm_base[(i, j)] = u_real_baseline[(perm_u[i], perm_d[j])];
        }
    }
    let (t12_b, t13_b, t23_b) = extract_pmns_angles(&u_perm_base);
    println!("--- CP VIOLATION: PHASE-ONLY COMPLEXIFICATION ---");
    println!(
        "  Real baseline: theta_12={:.2}, theta_13={:.2}, theta_23={:.2}",
        t12_b, t13_b, t23_b
    );

    // Build friction profiles
    let (o1, o2, o3) = get_sedenion_subalgebras();
    let subs = [&o1, &o2, &o3];
    let sign_table = SignTableCache::new(16);
    let nu_a = MajoranaMode {
        gamma_index: nu_pair.0 - 1,
        cd_basis_index: nu_pair.0,
        cd_dim: 16,
    };
    let nu_b = MajoranaMode {
        gamma_index: nu_pair.1 - 1,
        cd_basis_index: nu_pair.1,
        cd_dim: 16,
    };

    let build_profile =
        |mode_i: &MajoranaMode, mode_j: &MajoranaMode, sub: &[usize]| -> [f64; 16] {
            let i = mode_i.cd_basis_index;
            let j = mode_j.cd_basis_index;
            let a_sparse = vec![(i, 1.0)];
            let a_rotated = rotate_sparse(&a_sparse, i, j, std::f64::consts::FRAC_PI_4);
            let b_sparse = vec![(j, 1.0)];
            let mut profile = [0.0_f64; 16];
            for &kk in sub {
                if kk == 0 || kk == i || kk == j {
                    continue;
                }
                let x_sparse = [(kk, 1.0)];
                profile[kk] = sign_table.sparse_associator_sum(&a_rotated, &x_sparse, &b_sparse);
            }
            profile
        };

    let nu_profiles: Vec<[f64; 16]> = subs
        .iter()
        .map(|s| build_profile(&nu_a, &nu_b, s))
        .collect();

    let dot16 =
        |a: &[f64; 16], b: &[f64; 16]| -> f64 { a.iter().zip(b.iter()).map(|(x, y)| x * y).sum() };

    println!("\n  Scanning k=1..7 embeddings with phase-only complexification:");

    // For each k=1..7, compute phase angles from J_k complex structure
    for k in 1..=7 {
        // Full 16D J_k action on both octonion halves (14D active)
        // instead of 6D perp-only action from ComplexStructure
        let apply_jk = |v: &[f64; 16]| -> [f64; 16] { apply_jk_full_16d(v, k) };

        // Compute BOTH real and imaginary psi-overlaps
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

        // Compute natural phase angles from the overlap structure
        let mut phi = [[0.0_f64; 3]; 3];
        let mut has_nonzero_phase = false;
        for i in 0..3 {
            for j in 0..3 {
                if i != j {
                    phi[i][j] = im_overlap[i][j].atan2(re_overlap[i][j]);
                    if phi[i][j].abs() > 1e-10 {
                        has_nonzero_phase = true;
                    }
                }
            }
        }

        if !has_nonzero_phase {
            println!("  k={}: all phases zero, skipping", k);
            continue;
        }

        // Scan alpha_CP in [0.001, 1.0] with phase-only modification:
        // M_nu[i][j] -> |M_nu[i][j]| * exp(i * alpha_CP * phi[i][j])
        let mut best_alpha_cp = 0.0_f64;
        let mut best_j_cp = 0.0_f64;
        let mut best_delta = 0.0_f64;
        let mut best_score = f64::MAX;
        let mut best_angles = (0.0_f64, 0.0_f64, 0.0_f64);

        for step in 1..=200_i32 {
            let alpha_cp = step as f64 * 0.005;

            // Build Hermitian complex neutrino mass matrix
            // M[i][j] = M_real[i][j] * exp(i * alpha_CP * phi[i][j])
            // with Hermiticity: M[j][i] = conj(M[i][j])
            let mut m_nu_c = faer::Mat::<faer::c64>::zeros(3, 3);
            let mut m_ch_c = faer::Mat::<faer::c64>::zeros(3, 3);

            for i in 0..3 {
                // Diagonal stays real
                m_nu_c[(i, i)] = faer::c64::new(m_nu_corrected[(i, i)], 0.0);
                m_ch_c[(i, i)] = faer::c64::new(m_ch_real[(i, i)], 0.0);

                for j in (i + 1)..3 {
                    // Off-diagonal: phase rotation
                    let phase = alpha_cp * phi[i][j];
                    let mag = m_nu_corrected[(i, j)];
                    let re = mag * phase.cos();
                    let im = mag * phase.sin();
                    m_nu_c[(i, j)] = faer::c64::new(re, im);
                    m_nu_c[(j, i)] = faer::c64::new(re, -im); // Hermitian

                    // Charged lepton stays real symmetric
                    m_ch_c[(i, j)] = faer::c64::new(m_ch_real[(i, j)], 0.0);
                    m_ch_c[(j, i)] = faer::c64::new(m_ch_real[(j, i)], 0.0);
                }
            }

            let eig_ch_c = m_ch_c.self_adjoint_eigen(faer::Side::Lower).unwrap();
            let eig_nu_c = m_nu_c.self_adjoint_eigen(faer::Side::Lower).unwrap();
            let u_pmns_c = eig_ch_c.U().adjoint() * eig_nu_c.U();

            // Apply same permutation as real baseline
            let mut u_perm_c = faer::Mat::<faer::c64>::zeros(3, 3);
            for i in 0..3 {
                for j in 0..3 {
                    u_perm_c[(i, j)] = u_pmns_c[(perm_u[i], perm_d[j])];
                }
            }

            // Extract angles from |U_ij|
            let u_e3_abs = u_perm_c[(0, 2)].abs();
            let theta_13 = u_e3_abs.min(1.0).asin().to_degrees();
            let cos_13 = theta_13.to_radians().cos();
            let theta_12 = if cos_13 > 1e-15 {
                (u_perm_c[(0, 1)].abs() / cos_13)
                    .min(1.0)
                    .asin()
                    .to_degrees()
            } else {
                0.0
            };
            let theta_23 = if cos_13 > 1e-15 {
                (u_perm_c[(1, 2)].abs() / cos_13)
                    .min(1.0)
                    .asin()
                    .to_degrees()
            } else {
                0.0
            };

            // Jarlskog invariant: J = Im(U_e1 * U_mu2 * conj(U_e2) * conj(U_mu1))
            let j_cp = (u_perm_c[(0, 0)]
                * u_perm_c[(1, 1)]
                * u_perm_c[(0, 1)].conj()
                * u_perm_c[(1, 0)].conj())
            .im;

            // delta_CP from arg(-U_e3)
            let delta_cp = (-u_perm_c[(0, 2)]).arg().to_degrees();

            // Score: angle preservation is primary, J_CP is secondary reward
            let angle_cost = ((theta_12 - 33.41) / 33.41).powi(2)
                + ((theta_13 - 8.54) / 8.54).powi(2)
                + ((theta_23 - 49.0) / 49.0).powi(2);

            // Only accept if angles are within 5% of PDG
            if angle_cost < 0.01 && j_cp.abs() > 1e-6 {
                let score = angle_cost - 0.1 * j_cp.abs();
                if score < best_score {
                    best_score = score;
                    best_alpha_cp = alpha_cp;
                    best_j_cp = j_cp;
                    best_delta = delta_cp;
                    best_angles = (theta_12, theta_13, theta_23);
                }
            }
        }

        if best_j_cp.abs() > 1e-6 {
            let err_12 = ((best_angles.0 - 33.41) / 33.41 * 100.0).abs();
            let err_13 = ((best_angles.1 - 8.54) / 8.54 * 100.0).abs();
            let err_23 = ((best_angles.2 - 49.0) / 49.0 * 100.0).abs();
            println!(
                "  k={}: alpha_CP={:.3}, theta_12={:.2} ({:.1}%), theta_13={:.2} ({:.1}%), theta_23={:.2} ({:.1}%), J_CP={:.4e}, delta={:.1} deg",
                k,
                best_alpha_cp,
                best_angles.0,
                err_12,
                best_angles.1,
                err_13,
                best_angles.2,
                err_23,
                best_j_cp,
                best_delta
            );
        } else {
            println!(
                "  k={}: no solution with <5% angle error and nonzero J_CP",
                k
            );
        }
    }

    println!("\n  PDG targets: J_CP ~ 3.3e-2, delta_CP ~ 195 deg (normal ordering, NuFIT 5.3)");
    println!(
        "  Rephasing-invariant Jarlskog: |J| = cos(t12)*sin(t12)*cos(t13)^2*sin(t13)*cos(t23)*sin(t23)*sin(delta)"
    );
    println!("  With PDG angles + delta=195: J ~ -0.033");
}

/// Diagnostic: compare 6D (perp-only) vs full 16D J_k action for
/// CP violation across all seven octonion embeddings (C-1496).
///
/// # Purpose
///
/// Tests whether the |J_CP| gap to PDG (8.5e-3 vs 3.3e-2) is caused
/// by truncating the J_k action to 6 perpendicular indices, or whether
/// the gap is intrinsic to the algebraic structure of the friction
/// profiles.  This is the key architectural question from the plan.
///
/// # Two J_k implementations compared
///
/// ```text
/// 6D perp-only:  ComplexStructure::matrix[6][6] on perp_indices
///                (from g2_stabilizer::complex_structure)
///                Active dimensions: 6 (e_k^perp within lower octonion)
///
/// 16D full:      apply_jk_full_16d() -- e_k left-multiplication
///                on both octonion halves independently
///                Active dimensions: 14 (7+7 imaginary, minus k and k+8)
/// ```
///
/// # Methodology
///
/// For each k=1..7, builds the same friction profiles, computes phase
/// angles `phi[i][j]` using each J_k variant, then runs the full
/// phase-only CP pipeline (alpha_CP scan in [0.005, 1.0], Hermitian
/// eigendecomposition, Jarlskog extraction).  Reports the best
/// (alpha_CP, angles, |J_CP|, delta_CP) for each variant side by side.
///
/// # Expected result
///
/// NULL: both variants produce nearly identical |J_CP| (within 3%).
/// This is because friction profiles from selectors (e_7, e_8) have
/// zero components in the upper octonion block (indices 8-15), so the
/// 16D upper-block multiplication acts on zeros.
///
/// # Output format
///
/// ```text
/// k | dim | alpha_CP | t12    t13    t23   | |J_CP|     delta
/// --+-----+----------+-------...
/// 1 |  6D | 0.0500   | 33.35  8.66  48.93  | 8.26e-3    165.9
/// 1 | 16D | 0.0600   | 33.35  8.66  48.93  | 8.36e-3   -165.7
/// ```
///
/// # Claims exercised
///
/// - C-1496: 16D vs 6D J_k null result
/// - C-1494: Phase-only baseline (both variants reproduce it)
#[test]
fn test_cp_violation_jk_dimension_comparison() {
    use crate::{
        bell_inequality::{SignTableCache, rotate_sparse},
        majorana_braiding::MajoranaMode,
        three_fermion_generations::get_sedenion_subalgebras,
    };
    use cd_kernel::gourlay_psi;
    use gororoba_algebra::lie::g2_stabilizer::complex_structure;

    let ch_pair = (11_usize, 12);
    let nu_pair = (7_usize, 8);
    let alpha_ch = 3.00_f64;
    let alpha_nu = 1.35_f64;

    let (m_ch_real, m_nu_real) =
        construct_pmns_matrices_two_param(ch_pair, nu_pair, alpha_ch, alpha_nu);
    let (v6_basis, _, _) = extract_v6_basis();
    let lift = TensorElementLift;
    let eps = 0.05_f64;
    let n_basis = v6_basis.nrows().min(6);

    let eig_ch_0 = m_ch_real.self_adjoint_eigen(faer::Side::Lower).unwrap();
    let eig_nu_0 = m_nu_real.self_adjoint_eigen(faer::Side::Lower).unwrap();
    let u_raw_0 = eig_ch_0.U().transpose() * eig_nu_0.U();
    let (_, perm_u, perm_d) = crate::quark_sector::extract_ckm_permutation_aware(&u_raw_0);

    let angles_at = |beta: &[f64; 6]| -> (f64, f64, f64) {
        let mut m_nu = m_nu_real.clone();
        apply_v6_perturbation(&mut m_nu, &v6_basis, beta, &lift);
        let m_nu_s = (&m_nu + m_nu.transpose()) * faer::Scale(0.5);
        let eig_nu = m_nu_s.self_adjoint_eigen(faer::Side::Lower).unwrap();
        let u_raw = eig_ch_0.U().transpose() * eig_nu.U();
        let mut u_perm = faer::Mat::zeros(3, 3);
        for i in 0..3 {
            for j in 0..3 {
                u_perm[(i, j)] = u_raw[(perm_u[i], perm_d[j])];
            }
        }
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

    let inner_angles = |t_sol: f64, t_atm: f64| -> (f64, f64, f64) {
        let mut beta = [0.0_f64; 6];
        for kk in 0..6 {
            beta[kk] = t_sol * u_solar[kk] + t_atm * u_atmo[kk];
        }
        angles_at(&beta)
    };
    let (t_sol, t_atm, _, _) = gauss_newton_2d(
        &inner_angles,
        1.5,
        0.0,
        (33.41, 8.54, 49.0),
        (1.0, 2.24, 1.0),
        15,
    );

    let mut beta_opt = [0.0_f64; 6];
    for kk in 0..6 {
        beta_opt[kk] = t_sol * u_solar[kk] + t_atm * u_atmo[kk];
    }
    let mut m_nu_corrected = m_nu_real.clone();
    apply_v6_perturbation(&mut m_nu_corrected, &v6_basis, &beta_opt, &lift);
    let m_nu_corrected = (&m_nu_corrected + m_nu_corrected.transpose()) * faer::Scale(0.5);

    // Build friction profiles
    let (o1, o2, o3) = get_sedenion_subalgebras();
    let subs = [&o1, &o2, &o3];
    let sign_table = SignTableCache::new(16);
    let nu_a = MajoranaMode {
        gamma_index: nu_pair.0 - 1,
        cd_basis_index: nu_pair.0,
        cd_dim: 16,
    };
    let nu_b = MajoranaMode {
        gamma_index: nu_pair.1 - 1,
        cd_basis_index: nu_pair.1,
        cd_dim: 16,
    };

    let build_profile =
        |mode_i: &MajoranaMode, mode_j: &MajoranaMode, sub: &[usize]| -> [f64; 16] {
            let i = mode_i.cd_basis_index;
            let j = mode_j.cd_basis_index;
            let a_sparse = vec![(i, 1.0)];
            let a_rotated = rotate_sparse(&a_sparse, i, j, std::f64::consts::FRAC_PI_4);
            let b_sparse = vec![(j, 1.0)];
            let mut profile = [0.0_f64; 16];
            for &kk in sub {
                if kk == 0 || kk == i || kk == j {
                    continue;
                }
                let x_sparse = [(kk, 1.0)];
                profile[kk] = sign_table.sparse_associator_sum(&a_rotated, &x_sparse, &b_sparse);
            }
            profile
        };

    let nu_profiles: Vec<[f64; 16]> = subs
        .iter()
        .map(|s| build_profile(&nu_a, &nu_b, s))
        .collect();

    let dot16 =
        |a: &[f64; 16], b: &[f64; 16]| -> f64 { a.iter().zip(b.iter()).map(|(x, y)| x * y).sum() };

    println!("--- CP VIOLATION: 6D vs 16D J_k DIMENSION COMPARISON ---\n");
    println!(
        "  {:>2} | {:>5} | {:>8} | {:>8} {:>8} {:>8} | {:>10} {:>8} | {:>10} {:>8}",
        "k", "dim", "alpha_CP", "t12", "t13", "t23", "|J_CP|", "delta", "|J_CP|_16D", "delta_16D"
    );
    println!(
        "  {:-<2}-+-{:-<5}-+-{:-<8}-+-{:-<8}-{:-<8}-{:-<8}-+-{:-<10}-{:-<8}-+-{:-<10}-{:-<8}",
        "", "", "", "", "", "", "", "", "", ""
    );

    for k in 1..=7 {
        let cs = complex_structure(k);

        // 6D perp-only action
        let apply_jk_6d = |v: &[f64; 16]| -> [f64; 16] {
            let mut result = [0.0_f64; 16];
            for r in 0..6 {
                for s in 0..6 {
                    result[cs.perp_indices[r]] += cs.matrix[r][s] * v[cs.perp_indices[s]];
                }
            }
            result
        };

        // Full 16D action
        let apply_jk_16d = |v: &[f64; 16]| -> [f64; 16] { apply_jk_full_16d(v, k) };

        // Generic CP pipeline: takes a J_k action as a trait object and
        // returns (best_alpha, t12, t13, t23, j_cp, delta_cp).
        // Uses dyn Fn so the same closure works for both 6D and 16D.
        let compute_cp_for_jk =
            |apply_fn: &dyn Fn(&[f64; 16]) -> [f64; 16]| -> (f64, f64, f64, f64, f64, f64) {
                let mut re_overlap = [[0.0_f64; 3]; 3];
                let mut im_overlap = [[0.0_f64; 3]; 3];
                for i in 0..3 {
                    for j in 0..3 {
                        let psi_j = gourlay_psi(&nu_profiles[j]);
                        re_overlap[i][j] = dot16(&nu_profiles[i], &psi_j);
                        if i != j {
                            let jk_psi_j = apply_fn(&psi_j);
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
                            if phi[i][j].abs() > 1e-10 {
                                has_phase = true;
                            }
                        }
                    }
                }

                if !has_phase {
                    return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
                }

                let mut best_alpha = 0.0_f64;
                let mut best_jcp = 0.0_f64;
                let mut best_delta = 0.0_f64;
                let mut best_score = f64::MAX;
                let mut best_ang = (0.0_f64, 0.0_f64, 0.0_f64);

                for step in 1..=200_i32 {
                    let alpha_cp = step as f64 * 0.005;
                    let mut m_nu_c = faer::Mat::<faer::c64>::zeros(3, 3);
                    let mut m_ch_c = faer::Mat::<faer::c64>::zeros(3, 3);

                    for i in 0..3 {
                        m_nu_c[(i, i)] = faer::c64::new(m_nu_corrected[(i, i)], 0.0);
                        m_ch_c[(i, i)] = faer::c64::new(m_ch_real[(i, i)], 0.0);
                        for j in (i + 1)..3 {
                            let phase = alpha_cp * phi[i][j];
                            let mag = m_nu_corrected[(i, j)];
                            let re = mag * phase.cos();
                            let im = mag * phase.sin();
                            m_nu_c[(i, j)] = faer::c64::new(re, im);
                            m_nu_c[(j, i)] = faer::c64::new(re, -im);
                            m_ch_c[(i, j)] = faer::c64::new(m_ch_real[(i, j)], 0.0);
                            m_ch_c[(j, i)] = faer::c64::new(m_ch_real[(j, i)], 0.0);
                        }
                    }

                    let eig_ch_c = m_ch_c.self_adjoint_eigen(faer::Side::Lower).unwrap();
                    let eig_nu_c = m_nu_c.self_adjoint_eigen(faer::Side::Lower).unwrap();
                    let u_pmns_c = eig_ch_c.U().adjoint() * eig_nu_c.U();

                    let mut u_perm_c = faer::Mat::<faer::c64>::zeros(3, 3);
                    for i in 0..3 {
                        for j in 0..3 {
                            u_perm_c[(i, j)] = u_pmns_c[(perm_u[i], perm_d[j])];
                        }
                    }

                    let u_e3_abs = u_perm_c[(0, 2)].abs();
                    let theta_13 = u_e3_abs.min(1.0).asin().to_degrees();
                    let cos_13 = theta_13.to_radians().cos();
                    let theta_12 = if cos_13 > 1e-15 {
                        (u_perm_c[(0, 1)].abs() / cos_13)
                            .min(1.0)
                            .asin()
                            .to_degrees()
                    } else {
                        0.0
                    };
                    let theta_23 = if cos_13 > 1e-15 {
                        (u_perm_c[(1, 2)].abs() / cos_13)
                            .min(1.0)
                            .asin()
                            .to_degrees()
                    } else {
                        0.0
                    };

                    let j_cp = (u_perm_c[(0, 0)]
                        * u_perm_c[(1, 1)]
                        * u_perm_c[(0, 1)].conj()
                        * u_perm_c[(1, 0)].conj())
                    .im;

                    let delta_cp = (-u_perm_c[(0, 2)]).arg().to_degrees();

                    let angle_cost = ((theta_12 - 33.41) / 33.41).powi(2)
                        + ((theta_13 - 8.54) / 8.54).powi(2)
                        + ((theta_23 - 49.0) / 49.0).powi(2);

                    if angle_cost < 0.01 && j_cp.abs() > 1e-6 {
                        let score = angle_cost - 0.1 * j_cp.abs();
                        if score < best_score {
                            best_score = score;
                            best_alpha = alpha_cp;
                            best_jcp = j_cp;
                            best_delta = delta_cp;
                            best_ang = (theta_12, theta_13, theta_23);
                        }
                    }
                }

                (
                    best_alpha, best_ang.0, best_ang.1, best_ang.2, best_jcp, best_delta,
                )
            };

        let (a6, t12_6, t13_6, t23_6, jcp_6, d6) = compute_cp_for_jk(&apply_jk_6d);
        let (a16, t12_16, t13_16, t23_16, jcp_16, d16) = compute_cp_for_jk(&apply_jk_16d);

        if jcp_6.abs() > 1e-6 || jcp_16.abs() > 1e-6 {
            println!(
                "  k={} |  6D  | {:.4}   | {:.2}   {:.2}   {:.2}   | {:.4e}   {:.1}   | --         --",
                k, a6, t12_6, t13_6, t23_6, jcp_6, d6
            );
            println!(
                "  k={} | 16D  | {:.4}   | {:.2}   {:.2}   {:.2}   | --         --       | {:.4e}   {:.1}",
                k, a16, t12_16, t13_16, t23_16, jcp_16, d16
            );
        } else {
            println!(
                "  k={}: both variants have zero J_CP within 5% angle tolerance",
                k
            );
        }
    }

    println!("\n  PDG target: |J_CP| ~ 3.3e-2, delta_CP ~ 195 deg");
    println!("  Baseline (6D perp-only): |J_CP| ~ 8.5e-3 (C-1494)");
    println!("  If 16D > 6D, the gap is architectural, not algebraic.");
}
