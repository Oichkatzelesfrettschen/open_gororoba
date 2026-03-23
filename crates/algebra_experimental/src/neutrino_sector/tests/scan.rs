use super::super::*;

    /// Extract delta_CP from the standard parameterization.
    ///
    /// In the PDG parameterization, delta_CP appears ONLY in U_e3:
    ///   U_e3 = s_13 * exp(-i * delta)
    ///
    /// The Gram phase matrix phi_ij = arg(G_ij) gives the raw phase each
    /// PMNS element would carry. But 5 of the 9 phases are unphysical
    /// (removable by charged-lepton and neutrino rephasing). The single
    /// physical phase is:
    ///
    ///   delta_CP = phi_e1 + phi_mu3 - phi_e3 - phi_mu1
    ///            = arg(U_e1 * U_mu3 * conj(U_e3) * conj(U_mu1))
    ///
    /// This is the rephasing-invariant quartet (Jarlskog invariant phase).
    #[test]
    fn test_delta_cp_from_gram_quartet() {
        use cd_kernel::gourlay_psi;
        use crate::majorana_braiding::MajoranaMode;
        use crate::bell_inequality::rotate_sparse;

        let ch_pair = (11_usize, 12);
        let nu_pair = (7_usize, 8);

        let (o1, o2, o3, sign_table) = super::psi_setup();
        let subs = [&o1, &o2, &o3];

        let build_profile = |sel: (usize, usize), sub: &[usize]| -> [f64; 16] {
            let mode_i = MajoranaMode { gamma_index: sel.0 - 1, cd_basis_index: sel.0, cd_dim: 16 };
            let mode_j = MajoranaMode { gamma_index: sel.1 - 1, cd_basis_index: sel.1, cd_dim: 16 };
            let i = mode_i.cd_basis_index;
            let j = mode_j.cd_basis_index;
            let a_sparse = vec![(i, 1.0)];
            let a_rotated = rotate_sparse(&a_sparse, i, j, std::f64::consts::FRAC_PI_4);
            let b_sparse = vec![(j, 1.0)];
            let mut profile = [0.0_f64; 16];
            for &k in sub {
                if k == 0 || k == i || k == j { continue; }
                let x_sparse = [(k, 1.0)];
                profile[k] = sign_table.sparse_associator_sum(&a_rotated, &x_sparse, &b_sparse);
            }
            profile
        };

        let ch_profiles: Vec<[f64; 16]> = subs.iter()
            .map(|s| build_profile(ch_pair, s)).collect();
        let nu_profiles: Vec<[f64; 16]> = subs.iter()
            .map(|s| build_profile(nu_pair, s)).collect();

        let dot16 = |a: &[f64; 16], b: &[f64; 16]| -> f64 {
            a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
        };

        let omega_re = -0.5_f64;
        let omega_im = 3.0_f64.sqrt() / 2.0;

        // Compute full 3x3 Gram phase matrix
        let mut gram_phases = [[0.0_f64; 3]; 3];
        let mut gram_complex = [[(0.0_f64, 0.0_f64); 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                let c0 = dot16(&ch_profiles[i], &nu_profiles[j]);
                let psi1_j = gourlay_psi(&nu_profiles[j]);
                let c1 = dot16(&ch_profiles[i], &psi1_j);
                let psi2_j = cd_kernel::gourlay_psi_n(&nu_profiles[j], 2);
                let c2 = dot16(&ch_profiles[i], &psi2_j);
                let re = c0 + c1 * omega_re + c2 * omega_re;
                let im = c1 * omega_im - c2 * omega_im;
                gram_phases[i][j] = im.atan2(re);
                gram_complex[i][j] = (re, im);
            }
        }

        println!("  === Delta_CP from Rephasing-Invariant Quartet ===\n");
        println!("  Gram phases (degrees):");
        for i in 0..3 {
            println!("    [{:>8.2}, {:>8.2}, {:>8.2}]",
                gram_phases[i][0].to_degrees(),
                gram_phases[i][1].to_degrees(),
                gram_phases[i][2].to_degrees());
        }

        // The rephasing-invariant quartet:
        // delta = phi[0][0] + phi[1][2] - phi[0][2] - phi[1][0]
        // (indices: e=0, mu=1, tau=2; 1st=e-type, 2nd=mu-type, 3rd=tau-type)
        let delta_quartet = gram_phases[0][0] + gram_phases[1][2]
                          - gram_phases[0][2] - gram_phases[1][0];
        println!("\n  Rephasing-invariant quartet:");
        println!("  delta = phi[0][0] + phi[1][2] - phi[0][2] - phi[1][0]");
        println!("        = {:.2} + {:.2} - {:.2} - {:.2}",
            gram_phases[0][0].to_degrees(), gram_phases[1][2].to_degrees(),
            gram_phases[0][2].to_degrees(), gram_phases[1][0].to_degrees());
        println!("        = {:.2} deg", delta_quartet.to_degrees());

        // Also compute the complex quartet product:
        // Q = G[0][0] * G[1][2] * conj(G[0][2]) * conj(G[1][0])
        use num_complex::Complex;
        let g = |i: usize, j: usize| -> Complex<f64> {
            Complex::new(gram_complex[i][j].0, gram_complex[i][j].1)
        };
        let quartet = g(0, 0) * g(1, 2) * g(0, 2).conj() * g(1, 0).conj();
        println!("\n  Complex quartet: {:.4} + {:.4}i", quartet.re, quartet.im);
        println!("  arg(quartet) = {:.2} deg", quartet.arg().to_degrees());
        println!("  |quartet| = {:.4}", quartet.norm());

        // Try all 4 possible quartets (different index assignments)
        println!("\n  All rephasing-invariant quartets:");
        let quartets = [
            ((0, 0), (1, 2), (0, 2), (1, 0), "e1*mu3/e3*mu1"),
            ((0, 1), (1, 2), (0, 2), (1, 1), "e2*mu3/e3*mu2"),
            ((0, 0), (1, 1), (0, 1), (1, 0), "e1*mu2/e2*mu1"),
            ((0, 0), (2, 2), (0, 2), (2, 0), "e1*tau3/e3*tau1"),
        ];
        for &((a, b), (c, d), (e, f), (h, k), label) in &quartets {
            let q = g(a, b) * g(c, d) * g(e, f).conj() * g(h, k).conj();
            println!("  {}: arg = {:.2} deg, |Q| = {:.4}",
                label, q.arg().to_degrees(), q.norm());
        }

        println!("\n  PDG 2024: delta_CP = 195 deg (= -165 deg)");
        println!("  The quartet phase is the algebraic prediction for delta_CP.");
    }

    /// Scan all 6 generation-to-flavor assignments for delta_CP.
    ///
    /// The algebra produces 3 generation indices (O_1, O_2, O_3) but the
    /// mapping to physical flavors (e, mu, tau) is a discrete choice with
    /// 3! = 6 possibilities. Each gives a different delta_CP from the
    /// rephasing-invariant quartet.
    #[test]
    fn test_delta_cp_all_flavor_assignments() {
        use cd_kernel::gourlay_psi;
        use crate::majorana_braiding::MajoranaMode;
        use crate::bell_inequality::rotate_sparse;
        use num_complex::Complex;

        let ch_pair = (11_usize, 12);
        let nu_pair = (7_usize, 8);

        let (o1, o2, o3, sign_table) = super::psi_setup();
        let subs = [&o1, &o2, &o3];

        let build_profile = |sel: (usize, usize), sub: &[usize]| -> [f64; 16] {
            let mode_i = MajoranaMode { gamma_index: sel.0 - 1, cd_basis_index: sel.0, cd_dim: 16 };
            let mode_j = MajoranaMode { gamma_index: sel.1 - 1, cd_basis_index: sel.1, cd_dim: 16 };
            let i = mode_i.cd_basis_index;
            let j = mode_j.cd_basis_index;
            let a_sparse = vec![(i, 1.0)];
            let a_rotated = rotate_sparse(&a_sparse, i, j, std::f64::consts::FRAC_PI_4);
            let b_sparse = vec![(j, 1.0)];
            let mut profile = [0.0_f64; 16];
            for &k in sub {
                if k == 0 || k == i || k == j { continue; }
                let x_sparse = [(k, 1.0)];
                profile[k] = sign_table.sparse_associator_sum(&a_rotated, &x_sparse, &b_sparse);
            }
            profile
        };

        let ch_profiles: Vec<[f64; 16]> = subs.iter()
            .map(|s| build_profile(ch_pair, s)).collect();
        let nu_profiles: Vec<[f64; 16]> = subs.iter()
            .map(|s| build_profile(nu_pair, s)).collect();

        let dot16 = |a: &[f64; 16], b: &[f64; 16]| -> f64 {
            a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
        };

        let omega_re = -0.5_f64;
        let omega_im = 3.0_f64.sqrt() / 2.0;

        // Full 3x3 complex Gram matrix
        let mut gc = [[Complex::new(0.0, 0.0); 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                let c0 = dot16(&ch_profiles[i], &nu_profiles[j]);
                let psi1_j = gourlay_psi(&nu_profiles[j]);
                let c1 = dot16(&ch_profiles[i], &psi1_j);
                let psi2_j = cd_kernel::gourlay_psi_n(&nu_profiles[j], 2);
                let c2 = dot16(&ch_profiles[i], &psi2_j);
                let re = c0 + c1 * omega_re + c2 * omega_re;
                let im = c1 * omega_im - c2 * omega_im;
                gc[i][j] = Complex::new(re, im);
            }
        }

        // All 6 permutations of (0,1,2) -> (e, mu, tau)
        let perms: [(usize, usize, usize, &str); 6] = [
            (0, 1, 2, "O1=e, O2=mu, O3=tau"),
            (0, 2, 1, "O1=e, O2=tau, O3=mu"),
            (1, 0, 2, "O1=mu, O2=e, O3=tau"),
            (1, 2, 0, "O1=mu, O2=tau, O3=e"),
            (2, 0, 1, "O1=tau, O2=e, O3=mu"),
            (2, 1, 0, "O1=tau, O2=mu, O3=e"),
        ];

        println!("  === Delta_CP: All 6 Generation-to-Flavor Assignments ===\n");
        println!("  {:>35} | {:>10} | {:>10}",
            "Assignment", "delta_CP", "Residual");
        println!("  {:-<35}-+-{:-<10}-+-{:-<10}", "", "", "");

        let pdg_delta = -165.0_f64; // 195 deg = -165 deg in [-180, 180]

        for &(e, mu, _tau, label) in &perms {
            // Quartet: G[e,0] * G[mu,2] * conj(G[e,2]) * conj(G[mu,0])
            // In the permuted indices: e-row=perm[e], mu-row=perm[mu]
            // Column indices 0,1,2 = mass eigenstates (no permutation)
            let q = gc[e][0] * gc[mu][2] * gc[e][2].conj() * gc[mu][0].conj();
            let delta = q.arg().to_degrees();
            let residual = ((delta - pdg_delta + 540.0) % 360.0) - 180.0;

            println!("  {:>35} | {:>+10.2} | {:>+10.2}",
                label, delta, residual);
        }

        // Also try with column permutations (mass eigenstate relabeling)
        println!("\n  With mass-eigenstate relabeling (column perm):");
        let col_perms: [(usize, usize, usize); 6] = [
            (0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0),
        ];

        let mut best_residual = 180.0_f64;
        let mut best_assignment = String::new();
        let mut best_delta = 0.0_f64;

        for &(e, mu, _tau, row_label) in &perms {
            for &(c1, c2, c3) in &col_perms {
                let cols = [c1, c2, c3];
                let q = gc[e][cols[0]] * gc[mu][cols[2]]
                      * gc[e][cols[2]].conj() * gc[mu][cols[0]].conj();
                let delta = q.arg().to_degrees();
                let residual = ((delta - pdg_delta + 540.0) % 360.0) - 180.0;

                if residual.abs() < best_residual.abs() {
                    best_residual = residual;
                    best_delta = delta;
                    best_assignment = format!("{} | cols=({},{},{})", row_label, c1, c2, c3);
                }
            }
        }

        println!("\n  === BEST MATCH ===");
        println!("  Assignment: {}", best_assignment);
        println!("  delta_CP = {:.2} deg", best_delta);
        println!("  PDG: {:.2} deg", pdg_delta);
        println!("  Residual: {:.2} deg", best_residual);
    }

    /// Full delta_CP with both charged-lepton AND neutrino Gram matrices.
    ///
    /// The PMNS matrix is U = U_ch^dagger * U_nu. The CP phase comes from
    /// the RELATIVE complex structure between both sectors, not just the
    /// neutrino sector. We compute:
    ///   G_ch[i][j] = sum_k omega^k * <ch_i, psi^k(ch_j)>  (intra-sector)
    ///   G_nu[i][j] = sum_k omega^k * <nu_i, psi^k(nu_j)>  (intra-sector)
    ///   G_cross[i][j] = sum_k omega^k * <ch_i, psi^k(nu_j)>  (cross-sector)
    ///
    /// The physical delta_CP involves the PRODUCT of the charged and neutrino
    /// rephasing: delta = arg(quartet from G_cross) - arg(quartet from G_ch)
    ///                    + arg(quartet from G_nu)
    ///
    /// Also scan alpha_ch and alpha_nu for sensitivity.
    #[test]
    fn test_delta_cp_full_bilateral() {
        use cd_kernel::gourlay_psi;
        use crate::majorana_braiding::MajoranaMode;
        use crate::bell_inequality::rotate_sparse;
        use num_complex::Complex;

        let ch_pair = (11_usize, 12);
        let nu_pair = (7_usize, 8);

        let (o1, o2, o3, sign_table) = super::psi_setup();
        let subs = [&o1, &o2, &o3];

        let build_profile = |sel: (usize, usize), sub: &[usize]| -> [f64; 16] {
            let mode_i = MajoranaMode { gamma_index: sel.0 - 1, cd_basis_index: sel.0, cd_dim: 16 };
            let mode_j = MajoranaMode { gamma_index: sel.1 - 1, cd_basis_index: sel.1, cd_dim: 16 };
            let i = mode_i.cd_basis_index;
            let j = mode_j.cd_basis_index;
            let a_sparse = vec![(i, 1.0)];
            let a_rotated = rotate_sparse(&a_sparse, i, j, std::f64::consts::FRAC_PI_4);
            let b_sparse = vec![(j, 1.0)];
            let mut profile = [0.0_f64; 16];
            for &k in sub {
                if k == 0 || k == i || k == j { continue; }
                let x_sparse = [(k, 1.0)];
                profile[k] = sign_table.sparse_associator_sum(&a_rotated, &x_sparse, &b_sparse);
            }
            profile
        };

        let ch_profiles: Vec<[f64; 16]> = subs.iter()
            .map(|s| build_profile(ch_pair, s)).collect();
        let nu_profiles: Vec<[f64; 16]> = subs.iter()
            .map(|s| build_profile(nu_pair, s)).collect();

        let dot16 = |a: &[f64; 16], b: &[f64; 16]| -> f64 {
            a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
        };

        let omega_re = -0.5_f64;
        let omega_im = 3.0_f64.sqrt() / 2.0;

        // Build all three Gram matrices
        let build_gram = |profiles_a: &[[f64; 16]], profiles_b: &[[f64; 16]]| -> [[Complex<f64>; 3]; 3] {
            let mut g = [[Complex::new(0.0, 0.0); 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    let c0 = dot16(&profiles_a[i], &profiles_b[j]);
                    let psi1_j = gourlay_psi(&profiles_b[j]);
                    let c1 = dot16(&profiles_a[i], &psi1_j);
                    let psi2_j = cd_kernel::gourlay_psi_n(&profiles_b[j], 2);
                    let c2 = dot16(&profiles_a[i], &psi2_j);
                    let re = c0 + c1 * omega_re + c2 * omega_re;
                    let im = c1 * omega_im - c2 * omega_im;
                    g[i][j] = Complex::new(re, im);
                }
            }
            g
        };

        let g_ch = build_gram(&ch_profiles, &ch_profiles);
        let g_nu = build_gram(&nu_profiles, &nu_profiles);
        let g_cross = build_gram(&ch_profiles, &nu_profiles);

        // Compute quartet for each Gram matrix
        let quartet = |g: &[[Complex<f64>; 3]; 3], e: usize, mu: usize, c1: usize, c3: usize| -> Complex<f64> {
            g[e][c1] * g[mu][c3] * g[e][c3].conj() * g[mu][c1].conj()
        };

        println!("  === Full Bilateral Delta_CP ===\n");

        // Check intra-sector phases (should be zero from earlier null result)
        let q_ch = quartet(&g_ch, 0, 1, 0, 2);
        let q_nu = quartet(&g_nu, 0, 1, 0, 2);
        let q_cross = quartet(&g_cross, 0, 1, 0, 2);
        println!("  Intra-charged quartet:  arg = {:.2} deg, |Q| = {:.4}", q_ch.arg().to_degrees(), q_ch.norm());
        println!("  Intra-neutrino quartet: arg = {:.2} deg, |Q| = {:.4}", q_nu.arg().to_degrees(), q_nu.norm());
        println!("  Cross-sector quartet:   arg = {:.2} deg, |Q| = {:.4}", q_cross.arg().to_degrees(), q_cross.norm());

        // The physical delta_CP for U = U_ch^dagger * U_nu involves:
        // delta = arg(Q_cross) because U_ch is real (Q_ch has arg=0) and
        // U_nu is real (Q_nu has arg=0). The cross-sector Gram carries
        // the entire CP phase.
        //
        // But if we use DIFFERENT psi coupling strengths for ch vs nu,
        // the effective Gram involves weighted profiles. Let's check
        // with the psi-coupled profiles.

        // With psi coupling: profile_i_coupled = profile_i + alpha * psi(profile_i)
        let build_coupled = |profiles: &[[f64; 16]], alpha: f64| -> Vec<[f64; 16]> {
            profiles.iter().map(|p| {
                let psi_p = gourlay_psi(p);
                let mut coupled = [0.0_f64; 16];
                for k in 0..16 {
                    coupled[k] = p[k] + alpha * psi_p[k];
                }
                coupled
            }).collect()
        };

        println!("\n  --- Alpha sensitivity scan ---");
        println!("  {:>8} {:>8} | {:>10} | {:>10}",
            "a_ch", "a_nu", "delta_CP", "residual");

        let pdg_delta = -165.0_f64;
        let mut best_residual = 180.0_f64;
        let mut best_params = (0.0_f64, 0.0_f64, 0.0_f64);

        // Scan alpha_ch x alpha_nu with best assignment (e=0, mu=1, cols=(0,2,1))
        for ach_step in 0..=20 {
            let a_ch = ach_step as f64 * 0.5;
            for anu_step in 0..=20 {
                let a_nu = anu_step as f64 * 0.5;

                let ch_coupled = build_coupled(&ch_profiles, a_ch);
                let nu_coupled = build_coupled(&nu_profiles, a_nu);
                let gc = build_gram(&ch_coupled, &nu_coupled);

                // Best assignment from previous test: e=0, mu=1, cols=(0,2,1)
                let q = gc[0][0] * gc[1][1] * gc[0][1].conj() * gc[1][0].conj();
                let delta = q.arg().to_degrees();
                let residual = ((delta - pdg_delta + 540.0) % 360.0) - 180.0;

                if residual.abs() < best_residual.abs() {
                    best_residual = residual;
                    best_params = (a_ch, a_nu, delta);
                }
            }
        }

        println!("\n  === BEST BILATERAL MATCH ===");
        println!("  alpha_ch = {:.1}, alpha_nu = {:.1}", best_params.0, best_params.1);
        println!("  delta_CP = {:.2} deg", best_params.2);
        println!("  PDG: {:.2} deg", pdg_delta);
        println!("  Residual: {:.2} deg", best_residual);

        // Also scan with the PMNS-optimized alpha values (3.75, 1.30)
        let ch_opt = build_coupled(&ch_profiles, 3.75);
        let nu_opt = build_coupled(&nu_profiles, 1.30);
        let gc_opt = build_gram(&ch_opt, &nu_opt);

        println!("\n  --- At PMNS-optimal alphas (3.75, 1.30) ---");
        // Scan all assignments + column perms at optimal alphas
        let perms: [(usize, usize, &str); 6] = [
            (0, 1, "e=O1,mu=O2"), (0, 2, "e=O1,mu=O3"),
            (1, 0, "e=O2,mu=O1"), (1, 2, "e=O2,mu=O3"),
            (2, 0, "e=O3,mu=O1"), (2, 1, "e=O3,mu=O2"),
        ];
        let col_perms: [(usize, usize); 6] = [
            (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1),
        ];

        let mut best2_residual = 180.0_f64;
        let mut best2_delta = 0.0_f64;
        let mut best2_label = String::new();

        for &(e, mu, label) in &perms {
            for &(c1, c3) in &col_perms {
                let q = gc_opt[e][c1] * gc_opt[mu][c3] * gc_opt[e][c3].conj() * gc_opt[mu][c1].conj();
                let delta = q.arg().to_degrees();
                let residual = ((delta - pdg_delta + 540.0) % 360.0) - 180.0;
                if residual.abs() < best2_residual.abs() {
                    best2_residual = residual;
                    best2_delta = delta;
                    best2_label = format!("{} cols=({},{})", label, c1, c3);
                }
            }
        }

        println!("  Best: {} -> delta_CP = {:.2} deg (residual {:.2})",
            best2_label, best2_delta, best2_residual);
    }

    /// Scan ALL selector pairs for delta_CP + mixing angle chi^2.
    ///
    /// For each (charged_pair, neutrino_pair), compute:
    /// 1. Mixing angles via the real PMNS pipeline (chi^2 vs PDG)
    /// 2. Cross-sector Gram quartet phase (delta_CP prediction)
    /// 3. Combined score = chi2_angles + w_CP * (delta_CP - PDG)^2
    #[test]
    fn test_delta_cp_selector_scan() {
        use rayon::prelude::*;
        use cd_kernel::gourlay_psi;
        use crate::bell_inequality::{SignTableCache, rotate_sparse};
        use crate::three_fermion_generations::get_sedenion_subalgebras;
        use num_complex::Complex;

        let pdg = Pdg2024::default();
        let pdg_delta = -165.0_f64;

        let (o1, o2, o3) = get_sedenion_subalgebras();
        let subs_owned: Vec<Vec<usize>> = vec![o1.clone(), o2.clone(), o3.clone()];

        let pairs: Vec<(usize, usize)> = (1..=15)
            .flat_map(|a| ((a + 1)..=15).map(move |b| (a, b)))
            .collect();

        let omega_re = -0.5_f64;
        let omega_im = 3.0_f64.sqrt() / 2.0;

        // Parallel scan over selector pairs
        let results: Vec<_> = pairs.par_iter().flat_map_iter(|&ch| {
            let subs = &subs_owned;
            pairs.iter().map(move |&nu| {
                let sign_table = SignTableCache::new(16);

                let build_profile = |sel: (usize, usize), sub: &[usize]| -> [f64; 16] {
                    let i = sel.0;
                    let j = sel.1;
                    let a_sparse = vec![(i, 1.0)];
                    let a_rotated = rotate_sparse(&a_sparse, i, j, std::f64::consts::FRAC_PI_4);
                    let b_sparse = vec![(j, 1.0)];
                    let mut profile = [0.0_f64; 16];
                    for &k in sub {
                        if k == 0 || k == i || k == j { continue; }
                        let x_sparse = [(k, 1.0)];
                        profile[k] = sign_table.sparse_associator_sum(&a_rotated, &x_sparse, &b_sparse);
                    }
                    profile
                };

                let dot16 = |a: &[f64; 16], b: &[f64; 16]| -> f64 {
                    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
                };

                let ch_profiles: Vec<[f64; 16]> = subs.iter()
                    .map(|s| build_profile(ch, s)).collect();
                let nu_profiles: Vec<[f64; 16]> = subs.iter()
                    .map(|s| build_profile(nu, s)).collect();

                // Cross-sector Gram matrix
                let mut gc = [[Complex::new(0.0, 0.0); 3]; 3];
                for i in 0..3 {
                    for j in 0..3 {
                        let c0 = dot16(&ch_profiles[i], &nu_profiles[j]);
                        let psi1_j = gourlay_psi(&nu_profiles[j]);
                        let c1 = dot16(&ch_profiles[i], &psi1_j);
                        let psi2_j = cd_kernel::gourlay_psi_n(&nu_profiles[j], 2);
                        let c2 = dot16(&ch_profiles[i], &psi2_j);
                        gc[i][j] = Complex::new(
                            c0 + c1 * omega_re + c2 * omega_re,
                            c1 * omega_im - c2 * omega_im,
                        );
                    }
                }

                // Best delta_CP across assignments + column perms
                let mut best_delta_residual = 180.0_f64;
                let mut best_delta = 0.0_f64;
                for &(e, mu) in &[(0usize, 1usize), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)] {
                    for &(c1, c3) in &[(0usize, 1usize), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)] {
                        let q = gc[e][c1] * gc[mu][c3] * gc[e][c3].conj() * gc[mu][c1].conj();
                        if q.norm() < 1e-20 { continue; }
                        let delta = q.arg().to_degrees();
                        let residual = ((delta - pdg_delta + 540.0) % 360.0) - 180.0;
                        if residual.abs() < best_delta_residual.abs() {
                            best_delta_residual = residual;
                            best_delta = delta;
                        }
                    }
                }

                // Mixing angles (quick compute_pmns)
                let r = compute_pmns(ch, nu);
                let chi2 = chi_squared_pmns(&r, &pdg);

                (chi2, best_delta, best_delta_residual.abs(), ch, nu)
            })
        }).collect();

        // Sort by combined score: angle chi2 + CP residual weight
        let mut sorted = results;
        sorted.sort_by(|a, b| {
            let score_a = a.0 + 0.5 * a.2 * a.2; // chi2 + 0.5 * delta_residual^2
            let score_b = b.0 + 0.5 * b.2 * b.2;
            score_a.partial_cmp(&score_b).unwrap()
        });

        println!("  === Delta_CP + Angle Chi2 Selector Scan ===\n");
        println!("  {:>6} | {:>10} {:>10} | {:>8} | {:>8} | {:>8}",
            "rank", "charged", "neutrino", "chi2", "delta_CP", "|resid|");
        println!("  {:-<6}-+-{:-<10}-{:-<10}-+-{:-<8}-+-{:-<8}-+-{:-<8}", "", "", "", "", "", "");

        for (rank, entry) in sorted.iter().take(15).enumerate() {
            println!("  {:>6} | {:>10?} {:>10?} | {:>8.1} | {:>+8.1} | {:>8.1}",
                rank + 1, entry.3, entry.4, entry.0, entry.1, entry.2);
        }

        // Find the entry closest to PDG delta
        let best_cp = sorted.iter().min_by(|a, b| a.2.partial_cmp(&b.2).unwrap()).unwrap();
        println!("\n  Closest to PDG delta_CP:");
        println!("  charged={:?}, neutrino={:?}", best_cp.3, best_cp.4);
        println!("  chi2_angles = {:.1}, delta_CP = {:.1} deg, |residual| = {:.1} deg",
            best_cp.0, best_cp.1, best_cp.2);

        // Find the entry with best chi2 that also has |residual| < 60 deg
        let best_combined = sorted.iter()
            .filter(|e| e.2 < 60.0)
            .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        if let Some(bc) = best_combined {
            println!("\n  Best angles + decent CP (|resid| < 60 deg):");
            println!("  charged={:?}, neutrino={:?}", bc.3, bc.4);
            println!("  chi2_angles = {:.1}, delta_CP = {:.1} deg, |residual| = {:.1} deg",
                bc.0, bc.1, bc.2);
        }
    }

    /// Test the CP-optimal pair (11,13)/(11,14) with psi coupling.
    ///
    /// This pair gives delta_CP = -166 deg (1 deg from PDG) at baseline,
    /// but angles are far off. Apply the two-param psi coupling and scan
    /// alpha_ch x alpha_nu to see if angles can be brought closer.
    #[test]
    fn test_cp_optimal_pair_with_psi() {
        use rayon::prelude::*;

        let pdg = Pdg2024::default();

        // CP-optimal pairs to test
        let cp_pairs: [(usize, usize, usize, usize); 3] = [
            (11, 13, 11, 14),
            (9, 14, 9, 15),
            (10, 15, 10, 13),
        ];

        println!("  === CP-Optimal Pairs with Psi Coupling ===\n");

        for &(ch_a, ch_b, nu_a, nu_b) in &cp_pairs {
            let ch_pair = (ch_a, ch_b);
            let nu_pair = (nu_a, nu_b);

            println!("  --- Pair: ch=({},{}), nu=({},{}) ---", ch_a, ch_b, nu_a, nu_b);

            // Scan alpha_ch x alpha_nu
            let grid: Vec<(f64, f64)> = (0..=20)
                .flat_map(|a| (0..=20).map(move |b| (a as f64 * 0.5, b as f64 * 0.5)))
                .collect();

            let results: Vec<_> = grid.par_iter().map(|&(a_ch, a_nu)| {
                let (m_ch, m_nu) = construct_pmns_matrices_two_param(
                    ch_pair, nu_pair, a_ch, a_nu
                );
                let eig_ch = m_ch.selfadjoint_eigendecomposition(faer::Side::Lower);
                let eig_nu = m_nu.selfadjoint_eigendecomposition(faer::Side::Lower);
                let u_raw = eig_ch.u().transpose() * eig_nu.u();
                let (u_pmns, _, _) = crate::quark_sector::extract_ckm_permutation_aware(&u_raw);
                let (t12, t13, t23) = extract_pmns_angles(&u_pmns);
                let chi2 = ((t12 - pdg.theta_12_deg) / pdg.theta_12_err).powi(2)
                         + ((t13 - pdg.theta_13_deg) / pdg.theta_13_err).powi(2)
                         + ((t23 - pdg.theta_23_deg) / pdg.theta_23_err).powi(2);
                (chi2, a_ch, a_nu, t12, t13, t23)
            }).collect();

            let best = results.iter().min_by(|a, b| a.0.partial_cmp(&b.0).unwrap()).unwrap();
            println!("  Best: alpha_ch={:.1}, alpha_nu={:.1}", best.1, best.2);
            println!("  t12={:.2}, t13={:.2}, t23={:.2}, chi2={:.1}",
                best.3, best.4, best.5, best.0);
            println!("  PDG errors: t12={:.1}%, t13={:.1}%, t23={:.1}%",
                ((best.3 - pdg.theta_12_deg) / pdg.theta_12_deg * 100.0).abs(),
                ((best.4 - pdg.theta_13_deg) / pdg.theta_13_deg * 100.0).abs(),
                ((best.5 - pdg.theta_23_deg) / pdg.theta_23_deg * 100.0).abs());
            println!();
        }
    }

    /// Composite selector: blend angle-optimal and CP-optimal friction profiles.
    ///
    /// profile_blended = (1-w) * profile_angle_pair + w * profile_cp_pair
    ///
    /// Scan blend weight w to find the sweet spot where mixing angles remain
    /// close to PDG AND delta_CP moves toward -165 deg.
    #[test]
    fn test_composite_selector_blend() {
        use cd_kernel::gourlay_psi;
        use crate::bell_inequality::rotate_sparse;
        use num_complex::Complex;

        let pdg = Pdg2024::default();
        let pdg_delta = -165.0_f64;

        // Angle-optimal pair
        let angle_ch = (11_usize, 12);
        let angle_nu = (7_usize, 8);
        // CP-optimal pair
        let cp_ch = (11_usize, 13);
        let cp_nu = (11_usize, 14);

        let (o1, o2, o3, sign_table) = super::psi_setup();
        let subs = [&o1, &o2, &o3];

        let build_profile = |sel: (usize, usize), sub: &[usize]| -> [f64; 16] {
            let i = sel.0;
            let j = sel.1;
            let a_sparse = vec![(i, 1.0)];
            let a_rotated = rotate_sparse(&a_sparse, i, j, std::f64::consts::FRAC_PI_4);
            let b_sparse = vec![(j, 1.0)];
            let mut profile = [0.0_f64; 16];
            for &k in sub {
                if k == 0 || k == i || k == j { continue; }
                let x_sparse = [(k, 1.0)];
                profile[k] = sign_table.sparse_associator_sum(&a_rotated, &x_sparse, &b_sparse);
            }
            profile
        };

        // Build profiles for both pairs
        let angle_ch_profiles: Vec<[f64; 16]> = subs.iter()
            .map(|s| build_profile(angle_ch, s)).collect();
        let angle_nu_profiles: Vec<[f64; 16]> = subs.iter()
            .map(|s| build_profile(angle_nu, s)).collect();
        let cp_ch_profiles: Vec<[f64; 16]> = subs.iter()
            .map(|s| build_profile(cp_ch, s)).collect();
        let cp_nu_profiles: Vec<[f64; 16]> = subs.iter()
            .map(|s| build_profile(cp_nu, s)).collect();

        let dot16 = |a: &[f64; 16], b: &[f64; 16]| -> f64 {
            a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
        };

        let omega_re = -0.5_f64;
        let omega_im = 3.0_f64.sqrt() / 2.0;

        // First: check profile orthogonality
        println!("  === Composite Selector Blend ===\n");
        println!("  Profile orthogonality (gen 1):");
        let overlap_ch = dot16(&angle_ch_profiles[0], &cp_ch_profiles[0]);
        let norm_a_ch = dot16(&angle_ch_profiles[0], &angle_ch_profiles[0]).sqrt();
        let norm_c_ch = dot16(&cp_ch_profiles[0], &cp_ch_profiles[0]).sqrt();
        println!("    ch: cos(angle,cp) = {:.4} (norms: {:.2}, {:.2})",
            overlap_ch / (norm_a_ch * norm_c_ch), norm_a_ch, norm_c_ch);

        let overlap_nu = dot16(&angle_nu_profiles[0], &cp_nu_profiles[0]);
        let norm_a_nu = dot16(&angle_nu_profiles[0], &angle_nu_profiles[0]).sqrt();
        let norm_c_nu = dot16(&cp_nu_profiles[0], &cp_nu_profiles[0]).sqrt();
        println!("    nu: cos(angle,cp) = {:.4} (norms: {:.2}, {:.2})",
            overlap_nu / (norm_a_nu * norm_c_nu), norm_a_nu, norm_c_nu);

        // Blend and scan
        println!("\n  {:>6} | {:>8} {:>8} {:>8} | {:>8} | {:>8} | {:>8}",
            "w", "t12", "t13", "t23", "chi2", "delta_CP", "|resid|");
        println!("  {:-<6}-+-{:-<8}-{:-<8}-{:-<8}-+-{:-<8}-+-{:-<8}-+-{:-<8}",
            "", "", "", "", "", "", "");

        let mut best_combined = (f64::MAX, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);

        for w_step in 0..=20 {
            let w = w_step as f64 * 0.05;

            // Blend profiles
            let blend = |a: &[f64; 16], b: &[f64; 16]| -> [f64; 16] {
                let mut out = [0.0_f64; 16];
                for k in 0..16 {
                    out[k] = (1.0 - w) * a[k] + w * b[k];
                }
                out
            };

            let blended_ch: Vec<[f64; 16]> = (0..3).map(|i|
                blend(&angle_ch_profiles[i], &cp_ch_profiles[i])
            ).collect();
            let blended_nu: Vec<[f64; 16]> = (0..3).map(|i|
                blend(&angle_nu_profiles[i], &cp_nu_profiles[i])
            ).collect();

            // Compute cross-sector Gram for delta_CP
            let mut gc = [[Complex::new(0.0, 0.0); 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    let c0 = dot16(&blended_ch[i], &blended_nu[j]);
                    let psi1_j = gourlay_psi(&blended_nu[j]);
                    let c1 = dot16(&blended_ch[i], &psi1_j);
                    let psi2_j = cd_kernel::gourlay_psi_n(&blended_nu[j], 2);
                    let c2 = dot16(&blended_ch[i], &psi2_j);
                    gc[i][j] = Complex::new(
                        c0 + c1 * omega_re + c2 * omega_re,
                        c1 * omega_im - c2 * omega_im,
                    );
                }
            }

            // Best quartet phase (using best assignment from earlier scan)
            let mut best_delta = 0.0_f64;
            let mut best_resid = 180.0_f64;
            for &(e, mu) in &[(0usize, 1usize), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)] {
                for &(c1, c3) in &[(0usize, 1usize), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)] {
                    let q = gc[e][c1] * gc[mu][c3] * gc[e][c3].conj() * gc[mu][c1].conj();
                    if q.norm() < 1e-20 { continue; }
                    let delta = q.arg().to_degrees();
                    let resid = ((delta - pdg_delta + 540.0) % 360.0) - 180.0;
                    if resid.abs() < best_resid.abs() {
                        best_resid = resid;
                        best_delta = delta;
                    }
                }
            }

            // Compute mixing angles via blended diagonal friction
            // Use scalar friction from blended profiles as diagonal perturbation
            let sel_ch: Vec<f64> = blended_ch.iter().map(|p| {
                let norm: f64 = p.iter().map(|x| x * x).sum::<f64>();
                norm.sqrt()
            }).collect();
            let sel_nu: Vec<f64> = blended_nu.iter().map(|p| {
                let norm: f64 = p.iter().map(|x| x * x).sum::<f64>();
                norm.sqrt()
            }).collect();

            // Build mass matrices from blended friction
            let w1 = -0.656850_f64;
            let w2 = -0.741999_f64;
            let cb = construct_casimir_baseline(crate::quark_sector::SubalgebraScheme::InterleavedStride);
            let (m_base_ch, m_base_nu) = assemble_lepton_baseline(&cb);
            let mut m_ch = m_base_ch;
            let mut m_nu = m_base_nu;
            for i in 0..3 {
                let f_ch = w1 * sel_ch[i] + w2 * sel_nu[i];
                let f_nu = w1 * sel_nu[i] + w2 * sel_ch[i];
                m_ch.write(i, i, m_ch.read(i, i) + f_ch.exp());
                m_nu.write(i, i, m_nu.read(i, i) + f_nu.exp());
            }

            // Add psi off-diagonal with alpha = 3.75/1.30
            for i in 0..3 {
                for j in 0..3 {
                    if i == j { continue; }
                    let psi_nu_j = gourlay_psi(&blended_nu[j]);
                    let psi_ch_j = gourlay_psi(&blended_ch[j]);
                    m_nu.write(i, j, m_nu.read(i, j) + 1.30 * dot16(&blended_nu[i], &psi_nu_j));
                    m_ch.write(i, j, m_ch.read(i, j) + 3.75 * dot16(&blended_ch[i], &psi_ch_j));
                }
            }

            // Symmetrize + eigendecompose
            let m_ch_s = (&m_ch + m_ch.transpose()) * faer::scale(0.5);
            let m_nu_s = (&m_nu + m_nu.transpose()) * faer::scale(0.5);
            let eig_ch = m_ch_s.selfadjoint_eigendecomposition(faer::Side::Lower);
            let eig_nu = m_nu_s.selfadjoint_eigendecomposition(faer::Side::Lower);
            let u_raw = eig_ch.u().transpose() * eig_nu.u();
            let (u_pmns, _, _) = crate::quark_sector::extract_ckm_permutation_aware(&u_raw);
            let (t12, t13, t23) = extract_pmns_angles(&u_pmns);
            let chi2 = ((t12 - pdg.theta_12_deg) / pdg.theta_12_err).powi(2)
                     + ((t13 - pdg.theta_13_deg) / pdg.theta_13_err).powi(2)
                     + ((t23 - pdg.theta_23_deg) / pdg.theta_23_err).powi(2);

            // Combined score: angle chi2 + CP weight
            let score = chi2 + 0.1 * best_resid * best_resid;
            if score < best_combined.0 {
                best_combined = (score, w, t12, t13, t23, best_delta);
            }

            if w_step % 2 == 0 {
                println!("  {:>6.2} | {:>8.2} {:>8.2} {:>8.2} | {:>8.1} | {:>+8.1} | {:>8.1}",
                    w, t12, t13, t23, chi2, best_delta, best_resid.abs());
            }
        }

        println!("\n  === BEST COMBINED (chi2 + 0.1*|delta_resid|^2) ===");
        println!("  w = {:.2}", best_combined.1);
        println!("  t12 = {:.2}, t13 = {:.2}, t23 = {:.2}",
            best_combined.2, best_combined.3, best_combined.4);
        println!("  delta_CP = {:.1} deg", best_combined.5);
    }

    /// Split approach: angle-optimal PMNS + blended Gram rephasing.
    ///
    /// 1. Compute PMNS with angle-optimal selectors (11,12)/(7,8) at
    ///    Gauss-Newton parameters -> angles within 0.15% of PDG
    /// 2. Compute cross-sector Gram with blended profiles (w ~ 0.70)
    ///    -> delta_CP ~ -162 deg (2.9 deg from PDG)
    /// 3. Rephase the angle-optimal PMNS with the blended Gram phases
    /// 4. Report all 4 observables: 3 angles + delta_CP
    #[test]
    fn test_split_angle_cp() {
        use cd_kernel::gourlay_psi;
        use crate::bell_inequality::rotate_sparse;
        use num_complex::Complex;

        let pdg = Pdg2024::default();

        // === Step 1: Angle-optimal PMNS ===
        let angle_ch = (11_usize, 12);
        let angle_nu = (7_usize, 8);
        let (m_ch, m_nu) = construct_pmns_matrices_two_param(
            angle_ch, angle_nu, 3.75, 1.30
        );
        let eig_ch = m_ch.selfadjoint_eigendecomposition(faer::Side::Lower);
        let eig_nu = m_nu.selfadjoint_eigendecomposition(faer::Side::Lower);
        let u_raw = eig_ch.u().transpose() * eig_nu.u();
        let (u_real, _, _) = crate::quark_sector::extract_ckm_permutation_aware(&u_raw);
        let (t12, t13, t23) = extract_pmns_angles(&u_real);

        // === Step 2: Blended Gram phases ===
        let cp_ch = (11_usize, 13);
        let cp_nu = (11_usize, 14);

        let (o1, o2, o3, sign_table) = super::psi_setup();
        let subs = [&o1, &o2, &o3];

        let build_profile = |sel: (usize, usize), sub: &[usize]| -> [f64; 16] {
            let i = sel.0;
            let j = sel.1;
            let a_sparse = vec![(i, 1.0)];
            let a_rotated = rotate_sparse(&a_sparse, i, j, std::f64::consts::FRAC_PI_4);
            let b_sparse = vec![(j, 1.0)];
            let mut profile = [0.0_f64; 16];
            for &k in sub {
                if k == 0 || k == i || k == j { continue; }
                let x_sparse = [(k, 1.0)];
                profile[k] = sign_table.sparse_associator_sum(&a_rotated, &x_sparse, &b_sparse);
            }
            profile
        };

        let angle_ch_profiles: Vec<[f64; 16]> = subs.iter()
            .map(|s| build_profile(angle_ch, s)).collect();
        let angle_nu_profiles: Vec<[f64; 16]> = subs.iter()
            .map(|s| build_profile(angle_nu, s)).collect();
        let cp_ch_profiles: Vec<[f64; 16]> = subs.iter()
            .map(|s| build_profile(cp_ch, s)).collect();
        let cp_nu_profiles: Vec<[f64; 16]> = subs.iter()
            .map(|s| build_profile(cp_nu, s)).collect();

        let dot16 = |a: &[f64; 16], b: &[f64; 16]| -> f64 {
            a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
        };

        let omega_re = -0.5_f64;
        let omega_im = 3.0_f64.sqrt() / 2.0;

        println!("  === Split Approach: Angle-Optimal PMNS + Blended Gram CP ===\n");
        println!("  Angle-optimal PMNS (selectors (11,12)/(7,8), alpha 3.75/1.30):");
        println!("    t12={:.2}, t13={:.2}, t23={:.2}", t12, t13, t23);

        // Scan blend weight w for the Gram phases
        println!("\n  {:>6} | {:>8} {:>8} {:>8} | {:>10} | {:>+8} | {:>6}",
            "w", "t12", "t13", "t23", "J_CP", "delta", "|res|");
        println!("  {:-<6}-+-{:-<8}-{:-<8}-{:-<8}-+-{:-<10}-+-{:-<8}-+-{:-<6}",
            "", "", "", "", "", "", "");

        let mut best_w = 0.0_f64;
        let mut best_delta = 0.0_f64;
        let mut best_j = 0.0_f64;
        let mut best_resid = 180.0_f64;

        for w_step in 0..=40 {
            let w = w_step as f64 * 0.025;

            // Blend profiles for Gram computation only
            let blend = |a: &[f64; 16], b: &[f64; 16]| -> [f64; 16] {
                let mut out = [0.0_f64; 16];
                for k in 0..16 { out[k] = (1.0 - w) * a[k] + w * b[k]; }
                out
            };

            let blended_ch: Vec<[f64; 16]> = (0..3).map(|i|
                blend(&angle_ch_profiles[i], &cp_ch_profiles[i])
            ).collect();
            let blended_nu: Vec<[f64; 16]> = (0..3).map(|i|
                blend(&angle_nu_profiles[i], &cp_nu_profiles[i])
            ).collect();

            // Cross-sector Gram from blended profiles
            let mut gram_phases = [[0.0_f64; 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    let c0 = dot16(&blended_ch[i], &blended_nu[j]);
                    let psi1_j = gourlay_psi(&blended_nu[j]);
                    let c1 = dot16(&blended_ch[i], &psi1_j);
                    let psi2_j = cd_kernel::gourlay_psi_n(&blended_nu[j], 2);
                    let c2 = dot16(&blended_ch[i], &psi2_j);
                    let re = c0 + c1 * omega_re + c2 * omega_re;
                    let im = c1 * omega_im - c2 * omega_im;
                    gram_phases[i][j] = im.atan2(re);
                }
            }

            // === Step 3: Rephase the angle-optimal PMNS with blended Gram ===
            let mut u_cp = [[Complex::new(0.0, 0.0); 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    let mag = u_real.read(i, j).abs();
                    let sign = if u_real.read(i, j) >= 0.0 { 1.0 } else { -1.0 };
                    let phase = gram_phases[i][j];
                    u_cp[i][j] = Complex::from_polar(mag, phase) * sign;
                }
            }

            // Extract angles (preserved since |U_ij| unchanged)
            let u_e3_abs = u_cp[0][2].norm();
            let t13_cp = u_e3_abs.min(1.0).asin().to_degrees();
            let cos_13 = t13_cp.to_radians().cos();
            let t12_cp = if cos_13 > 1e-15 {
                (u_cp[0][1].norm() / cos_13).min(1.0).asin().to_degrees()
            } else { 0.0 };
            let t23_cp = if cos_13 > 1e-15 {
                (u_cp[1][2].norm() / cos_13).min(1.0).asin().to_degrees()
            } else { 0.0 };

            // Jarlskog
            let j_cp = (u_cp[0][1] * u_cp[1][2] * u_cp[0][2].conj() * u_cp[1][1].conj()).im;
            let delta = extract_cp_phase((t12_cp, t13_cp, t23_cp), j_cp);

            let pdg_delta = -165.0_f64;
            let resid = ((delta - pdg_delta + 540.0) % 360.0) - 180.0;

            if resid.abs() < best_resid.abs() {
                best_resid = resid;
                best_w = w;
                best_delta = delta;
                best_j = j_cp;
            }

            if w_step % 4 == 0 {
                println!("  {:>6.3} | {:>8.2} {:>8.2} {:>8.2} | {:>10.4e} | {:>+8.1} | {:>6.1}",
                    w, t12_cp, t13_cp, t23_cp, j_cp, delta, resid.abs());
            }
        }

        println!("\n  === SPLIT RESULT ===");
        println!("  Blend weight w = {:.3}", best_w);
        println!("  theta_12 = {:.2} deg (PDG: {:.2}, error: {:.2}%)",
            t12, pdg.theta_12_deg, ((t12 - pdg.theta_12_deg) / pdg.theta_12_deg * 100.0).abs());
        println!("  theta_13 = {:.2} deg (PDG: {:.2}, error: {:.2}%)",
            t13, pdg.theta_13_deg, ((t13 - pdg.theta_13_deg) / pdg.theta_13_deg * 100.0).abs());
        println!("  theta_23 = {:.2} deg (PDG: {:.2}, error: {:.2}%)",
            t23, pdg.theta_23_deg, ((t23 - pdg.theta_23_deg) / pdg.theta_23_deg * 100.0).abs());
        println!("  |J_CP| = {:.4e} (PDG: ~3e-2)", best_j.abs());
        println!("  delta_CP = {:.1} deg (PDG: -165, residual: {:.1} deg)", best_delta, best_resid.abs());
        println!("\n  ALL FOUR OBSERVABLES:");
        let chi2_full = ((t12 - pdg.theta_12_deg) / pdg.theta_12_err).powi(2)
                      + ((t13 - pdg.theta_13_deg) / pdg.theta_13_err).powi(2)
                      + ((t23 - pdg.theta_23_deg) / pdg.theta_23_err).powi(2)
                      + ((best_delta - (-165.0)) / pdg.delta_cp_err).powi(2);
        println!("  chi2 (3 angles + delta_CP) = {:.2} / 4 dof = {:.2}",
            chi2_full, chi2_full / 4.0);
    }

    /// 3-blade friction for mass ratio: use triple selectors instead of pairs.
    ///
    /// The 2-blade friction gives r = 0.1478 (4.8x PDG). The 3-blade friction
    /// (sum of 3 pairwise braid frictions) has a quantized spectrum in 2*sqrt(2)
    /// steps and can produce steeper hierarchies.
    #[test]
    fn test_3blade_mass_ratio() {
        use rayon::prelude::*;
        use crate::lepton_mass_hierarchy::cd_braid_signed_friction;
        use crate::majorana_braiding::MajoranaMode;
        use crate::bell_inequality::SignTableCache;
        use crate::three_fermion_generations::get_sedenion_subalgebras;

        let _pdg = Pdg2024::default();
        let pdg_r = 0.0307_f64;

        let (o1, o2, o3) = get_sedenion_subalgebras();
        let subs = [o1.clone(), o2.clone(), o3.clone()];

        // Generate all triples C(15,3) = 455
        let mut triples: Vec<(usize, usize, usize)> = Vec::new();
        for i in 1..16_usize {
            for j in (i + 1)..16 {
                for k in (j + 1)..16 {
                    triples.push((i, j, k));
                }
            }
        }

        let w1 = -0.656850_f64;
        let w2 = -0.741999_f64;

        // Scan all (ch_triple, nu_triple) combinations
        // For efficiency, precompute all 3-blade frictions first
        let compute_3blade = |triple: (usize, usize, usize)| -> [f64; 3] {
            let sign_table = SignTableCache::new(16);
            let (i, j, k) = triple;
            let mi = MajoranaMode { gamma_index: i - 1, cd_basis_index: i, cd_dim: 16 };
            let mj = MajoranaMode { gamma_index: j - 1, cd_basis_index: j, cd_dim: 16 };
            let mk = MajoranaMode { gamma_index: k - 1, cd_basis_index: k, cd_dim: 16 };
            let mut f = [0.0_f64; 3];
            for (g, sub) in subs.iter().enumerate() {
                f[g] = cd_braid_signed_friction(&mi, &mj, sub, &sign_table)
                     + cd_braid_signed_friction(&mi, &mk, sub, &sign_table)
                     + cd_braid_signed_friction(&mj, &mk, sub, &sign_table);
            }
            f
        };

        // Precompute all friction values
        let all_frictions: Vec<[f64; 3]> = triples.par_iter()
            .map(|&t| compute_3blade(t))
            .collect();

        // Now scan pairs of triples for mass ratio
        let af = &all_frictions;
        let results: Vec<_> = (0..triples.len()).into_par_iter().flat_map_iter(|ci| {
            (0..triples.len()).map(move |ni| {
                let sel_ch = &af[ci];
                let sel_nu = &af[ni];

                // Build diagonal mass contributions
                let mut masses = [0.0_f64; 3];
                for g in 0..3 {
                    let f = w1 * sel_ch[g] + w2 * sel_nu[g];
                    masses[g] = f.exp();
                }

                // Sort by magnitude
                masses.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let dm21 = masses[1] * masses[1] - masses[0] * masses[0];
                let dm31 = masses[2] * masses[2] - masses[0] * masses[0];
                let r = if dm31.abs() > 1e-30 { dm21 / dm31 } else { f64::MAX };
                let r_err = (r - pdg_r).abs();
                let hierarchy = masses[2] / masses[0];

                (r_err, r, hierarchy, ci, ni)
            })
        }).collect();

        let mut sorted = results;
        sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        println!("  === 3-Blade Mass Ratio Scan ===\n");
        println!("  PDG: r = {:.4}, m3/m1 ~ 50\n", pdg_r);
        println!("  {:>5} | {:>12} {:>12} | {:>8} | {:>6} | {:>8}",
            "rank", "ch_triple", "nu_triple", "r", "m3/m1", "|r_err|");
        println!("  {:-<5}-+-{:-<12}-{:-<12}-+-{:-<8}-+-{:-<6}-+-{:-<8}", "", "", "", "", "", "");

        for (rank, e) in sorted.iter().take(10).enumerate() {
            println!("  {:>5} | {:>12?} {:>12?} | {:>8.4} | {:>6.1} | {:>8.4}",
                rank + 1, triples[e.3], triples[e.4], e.1, e.2, e.0);
        }

        let best = &sorted[0];
        println!("\n  BEST: r = {:.6} (PDG: {:.4}, error: {:.1}%)",
            best.1, pdg_r, (best.1 - pdg_r) / pdg_r * 100.0);
        println!("  m3/m1 = {:.1}", best.2);
        println!("  ch_triple = {:?}", triples[best.3]);
        println!("  nu_triple = {:?}", triples[best.4]);
    }

    /// Unified 3-blade: mass ratio + mixing angles from the SAME triple selectors.
    ///
    /// Build full mass matrices using 3-blade friction (sum of 3 pairwise
    /// braid frictions) with psi off-diagonal coupling, then extract both
    /// the mass ratio r AND the mixing angles.
    #[test]
    fn test_3blade_unified() {
        use rayon::prelude::*;
        use cd_kernel::gourlay_psi;
        use crate::lepton_mass_hierarchy::cd_braid_signed_friction;
        use crate::majorana_braiding::MajoranaMode;
        use crate::bell_inequality::{SignTableCache, rotate_sparse};
        use crate::three_fermion_generations::get_sedenion_subalgebras;

        let pdg = Pdg2024::default();
        let pdg_r = 0.0307_f64;

        let (o1, o2, o3) = get_sedenion_subalgebras();
        let subs_owned = [o1.clone(), o2.clone(), o3.clone()];

        // Best 3-blade triples from the mass ratio scan
        let candidates: [(usize, usize, usize, usize, usize, usize); 3] = [
            (1, 6, 11, 1, 3, 8),     // r = 0.0304
            (4, 9, 15, 6, 7, 12),    // r = 0.0304
            (4, 11, 14, 5, 6, 12),   // r = 0.0304
        ];

        let w1 = -0.656850_f64;
        let w2 = -0.741999_f64;

        println!("  === Unified 3-Blade: Mass Ratio + Angles ===\n");

        for &(ci, cj, ck, ni, nj, nk) in &candidates {
            let sign_table = SignTableCache::new(16);
            let subs = &subs_owned;

            // Build 3-blade friction profiles (16D vectors)
            let build_3blade_profile = |a: usize, b: usize, c: usize, sub: &[usize]| -> [f64; 16] {
                let ma = MajoranaMode { gamma_index: a - 1, cd_basis_index: a, cd_dim: 16 };
                let mb = MajoranaMode { gamma_index: b - 1, cd_basis_index: b, cd_dim: 16 };
                let mc = MajoranaMode { gamma_index: c - 1, cd_basis_index: c, cd_dim: 16 };

                // Sum the 3 pairwise 16D profiles
                let build_pair_profile = |m1: &MajoranaMode, m2: &MajoranaMode, s: &[usize]| -> [f64; 16] {
                    let i = m1.cd_basis_index;
                    let j = m2.cd_basis_index;
                    let a_sparse = vec![(i, 1.0)];
                    let a_rotated = rotate_sparse(&a_sparse, i, j, std::f64::consts::FRAC_PI_4);
                    let b_sparse = vec![(j, 1.0)];
                    let mut profile = [0.0_f64; 16];
                    for &k in s {
                        if k == 0 || k == i || k == j { continue; }
                        let x_sparse = [(k, 1.0)];
                        profile[k] = sign_table.sparse_associator_sum(&a_rotated, &x_sparse, &b_sparse);
                    }
                    profile
                };

                let p_ab = build_pair_profile(&ma, &mb, sub);
                let p_ac = build_pair_profile(&ma, &mc, sub);
                let p_bc = build_pair_profile(&mb, &mc, sub);
                let mut combined = [0.0_f64; 16];
                for idx in 0..16 {
                    combined[idx] = p_ab[idx] + p_ac[idx] + p_bc[idx];
                }
                combined
            };

            let ch_profiles: Vec<[f64; 16]> = subs.iter()
                .map(|s| build_3blade_profile(ci, cj, ck, s)).collect();
            let nu_profiles: Vec<[f64; 16]> = subs.iter()
                .map(|s| build_3blade_profile(ni, nj, nk, s)).collect();

            // 3-blade scalar friction (for diagonal)
            let sel_ch: Vec<f64> = subs.iter().map(|s| {
                let ma = MajoranaMode { gamma_index: ci - 1, cd_basis_index: ci, cd_dim: 16 };
                let mb = MajoranaMode { gamma_index: cj - 1, cd_basis_index: cj, cd_dim: 16 };
                let mc = MajoranaMode { gamma_index: ck - 1, cd_basis_index: ck, cd_dim: 16 };
                cd_braid_signed_friction(&ma, &mb, s, &sign_table)
                + cd_braid_signed_friction(&ma, &mc, s, &sign_table)
                + cd_braid_signed_friction(&mb, &mc, s, &sign_table)
            }).collect();
            let sel_nu: Vec<f64> = subs.iter().map(|s| {
                let ma = MajoranaMode { gamma_index: ni - 1, cd_basis_index: ni, cd_dim: 16 };
                let mb = MajoranaMode { gamma_index: nj - 1, cd_basis_index: nj, cd_dim: 16 };
                let mc = MajoranaMode { gamma_index: nk - 1, cd_basis_index: nk, cd_dim: 16 };
                cd_braid_signed_friction(&ma, &mb, s, &sign_table)
                + cd_braid_signed_friction(&ma, &mc, s, &sign_table)
                + cd_braid_signed_friction(&mb, &mc, s, &sign_table)
            }).collect();

            let dot16 = |a: &[f64; 16], b: &[f64; 16]| -> f64 {
                a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
            };

            // Scan alpha_ch x alpha_nu
            let alpha_grid: Vec<(f64, f64)> = (0..=20)
                .flat_map(|a| (0..=20).map(move |b| (a as f64 * 0.5, b as f64 * 0.5)))
                .collect();

            let best = alpha_grid.par_iter().map(|&(a_ch, a_nu)| {
                let cb = construct_casimir_baseline(crate::quark_sector::SubalgebraScheme::InterleavedStride);
                let (m_base_ch, m_base_nu) = assemble_lepton_baseline(&cb);
                let mut m_ch = m_base_ch;
                let mut m_nu = m_base_nu;

                // Diagonal: 3-blade friction
                for g in 0..3 {
                    let f_ch = w1 * sel_ch[g] + w2 * sel_nu[g];
                    let f_nu = w1 * sel_nu[g] + w2 * sel_ch[g];
                    m_ch.write(g, g, m_ch.read(g, g) + f_ch.exp());
                    m_nu.write(g, g, m_nu.read(g, g) + f_nu.exp());
                }

                // Off-diagonal: psi coupling with 3-blade profiles
                for i in 0..3 {
                    for j in 0..3 {
                        if i == j { continue; }
                        let psi_nu_j = gourlay_psi(&nu_profiles[j]);
                        let psi_ch_j = gourlay_psi(&ch_profiles[j]);
                        m_nu.write(i, j, m_nu.read(i, j) + a_nu * dot16(&nu_profiles[i], &psi_nu_j));
                        m_ch.write(i, j, m_ch.read(i, j) + a_ch * dot16(&ch_profiles[i], &psi_ch_j));
                    }
                }

                // Symmetrize + eigendecompose
                let m_ch_s = (&m_ch + m_ch.transpose()) * faer::scale(0.5);
                let m_nu_s = (&m_nu + m_nu.transpose()) * faer::scale(0.5);
                let eig_ch = m_ch_s.selfadjoint_eigendecomposition(faer::Side::Lower);
                let eig_nu = m_nu_s.selfadjoint_eigendecomposition(faer::Side::Lower);
                let u_raw = eig_ch.u().transpose() * eig_nu.u();
                let (u_pmns, _, _) = crate::quark_sector::extract_ckm_permutation_aware(&u_raw);
                let (t12, t13, t23) = extract_pmns_angles(&u_pmns);

                // Mass ratio
                let mut ev: Vec<f64> = (0..3).map(|i| eig_nu.s().column_vector().read(i).abs()).collect();
                ev.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let dm21 = ev[1] * ev[1] - ev[0] * ev[0];
                let dm31 = ev[2] * ev[2] - ev[0] * ev[0];
                let r = if dm31.abs() > 1e-30 { dm21 / dm31 } else { f64::MAX };

                let chi2 = ((t12 - pdg.theta_12_deg) / pdg.theta_12_err).powi(2)
                         + ((t13 - pdg.theta_13_deg) / pdg.theta_13_err).powi(2)
                         + ((t23 - pdg.theta_23_deg) / pdg.theta_23_err).powi(2);
                let r_err = ((r - pdg_r) / pdg_r).powi(2);
                let score = chi2 + 100.0 * r_err; // weight mass ratio

                (score, a_ch, a_nu, t12, t13, t23, r, ev[2] / ev[0])
            }).min_by(|a, b| a.0.partial_cmp(&b.0).unwrap()).unwrap();

            println!("  ch=({},{},{}), nu=({},{},{})", ci, cj, ck, ni, nj, nk);
            println!("    alpha_ch={:.1}, alpha_nu={:.1}", best.1, best.2);
            println!("    t12={:.2}, t13={:.2}, t23={:.2}", best.3, best.4, best.5);
            println!("    r = {:.4} (PDG: {:.4}, err: {:.1}%)", best.6, pdg_r, (best.6 - pdg_r) / pdg_r * 100.0);
            println!("    m3/m1 = {:.1}", best.7);
            println!();
        }
    }

    /// Two-selector-type model: 3-blade diagonal + 2-blade off-diagonal.
    ///
    /// Diagonal mass hierarchy: 3-blade triples (1,6,11)/(1,3,8) -> r=0.0304
    /// Off-diagonal mixing: 2-blade pairs (11,12)/(7,8) -> angles ~0.15% PDG
    ///
    /// The model uses DIFFERENT selectors for diagonal (eigenvalue spacing)
    /// and off-diagonal (eigenvector rotation), analogous to the see-saw
    /// mechanism where heavy and light scales come from different physics.
    #[test]
    fn test_two_selector_type_model() {
        use cd_kernel::gourlay_psi;
        use crate::lepton_mass_hierarchy::cd_braid_signed_friction;
        use crate::majorana_braiding::MajoranaMode;
        use crate::bell_inequality::rotate_sparse;

        let pdg = Pdg2024::default();
        let pdg_r = 0.0307_f64;

        let (o1, o2, o3, sign_table) = super::psi_setup();
        let subs = [&o1, &o2, &o3];

        let w1 = -0.656850_f64;
        let w2 = -0.741999_f64;

        // === 3-blade diagonal friction (mass hierarchy) ===
        let ch_triple = (1_usize, 6, 11);
        let nu_triple = (1_usize, 3, 8);

        let sel_ch_3blade: Vec<f64> = subs.iter().map(|s| {
            let (a, b, c) = ch_triple;
            let ma = MajoranaMode { gamma_index: a - 1, cd_basis_index: a, cd_dim: 16 };
            let mb = MajoranaMode { gamma_index: b - 1, cd_basis_index: b, cd_dim: 16 };
            let mc = MajoranaMode { gamma_index: c - 1, cd_basis_index: c, cd_dim: 16 };
            cd_braid_signed_friction(&ma, &mb, s, &sign_table)
            + cd_braid_signed_friction(&ma, &mc, s, &sign_table)
            + cd_braid_signed_friction(&mb, &mc, s, &sign_table)
        }).collect();

        let sel_nu_3blade: Vec<f64> = subs.iter().map(|s| {
            let (a, b, c) = nu_triple;
            let ma = MajoranaMode { gamma_index: a - 1, cd_basis_index: a, cd_dim: 16 };
            let mb = MajoranaMode { gamma_index: b - 1, cd_basis_index: b, cd_dim: 16 };
            let mc = MajoranaMode { gamma_index: c - 1, cd_basis_index: c, cd_dim: 16 };
            cd_braid_signed_friction(&ma, &mb, s, &sign_table)
            + cd_braid_signed_friction(&ma, &mc, s, &sign_table)
            + cd_braid_signed_friction(&mb, &mc, s, &sign_table)
        }).collect();

        // === 2-blade off-diagonal profiles (mixing angles) ===
        let angle_ch = (11_usize, 12);
        let angle_nu = (7_usize, 8);

        let build_profile = |sel: (usize, usize), sub: &[usize]| -> [f64; 16] {
            let i = sel.0;
            let j = sel.1;
            let a_sparse = vec![(i, 1.0)];
            let a_rotated = rotate_sparse(&a_sparse, i, j, std::f64::consts::FRAC_PI_4);
            let b_sparse = vec![(j, 1.0)];
            let mut profile = [0.0_f64; 16];
            for &k in sub {
                if k == 0 || k == i || k == j { continue; }
                let x_sparse = [(k, 1.0)];
                profile[k] = sign_table.sparse_associator_sum(&a_rotated, &x_sparse, &b_sparse);
            }
            profile
        };

        let ch_profiles_2blade: Vec<[f64; 16]> = subs.iter()
            .map(|s| build_profile(angle_ch, s)).collect();
        let nu_profiles_2blade: Vec<[f64; 16]> = subs.iter()
            .map(|s| build_profile(angle_nu, s)).collect();

        let dot16 = |a: &[f64; 16], b: &[f64; 16]| -> f64 {
            a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
        };

        println!("  === Two-Selector-Type Model ===\n");
        println!("  Diagonal: 3-blade {:?}/{:?}", ch_triple, nu_triple);
        println!("  Off-diag: 2-blade {:?}/{:?}\n", angle_ch, angle_nu);

        println!("  3-blade frictions:");
        println!("    ch: [{:.2}, {:.2}, {:.2}]", sel_ch_3blade[0], sel_ch_3blade[1], sel_ch_3blade[2]);
        println!("    nu: [{:.2}, {:.2}, {:.2}]", sel_nu_3blade[0], sel_nu_3blade[1], sel_nu_3blade[2]);

        // Scan alpha_ch x alpha_nu for the 2-blade off-diagonal coupling
        println!("\n  {:>6} {:>6} | {:>8} {:>8} {:>8} | {:>8} | {:>6} | {:>6}",
            "a_ch", "a_nu", "t12", "t13", "t23", "r", "chi2", "score");

        let mut best_score = f64::MAX;
        let mut best_result = (0.0_f64, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

        for ach_step in 0..=20 {
            for anu_step in 0..=20 {
                let a_ch = ach_step as f64 * 0.5;
                let a_nu = anu_step as f64 * 0.5;

                let cb = construct_casimir_baseline(crate::quark_sector::SubalgebraScheme::InterleavedStride);
                let (m_base_ch, m_base_nu) = assemble_lepton_baseline(&cb);
                let mut m_ch = m_base_ch;
                let mut m_nu = m_base_nu;

                // Diagonal: 3-blade friction
                for g in 0..3 {
                    let f_ch = w1 * sel_ch_3blade[g] + w2 * sel_nu_3blade[g];
                    let f_nu = w1 * sel_nu_3blade[g] + w2 * sel_ch_3blade[g];
                    m_ch.write(g, g, m_ch.read(g, g) + f_ch.exp());
                    m_nu.write(g, g, m_nu.read(g, g) + f_nu.exp());
                }

                // Off-diagonal: 2-blade psi coupling
                for i in 0..3 {
                    for j in 0..3 {
                        if i == j { continue; }
                        let psi_nu_j = gourlay_psi(&nu_profiles_2blade[j]);
                        let psi_ch_j = gourlay_psi(&ch_profiles_2blade[j]);
                        m_nu.write(i, j, m_nu.read(i, j) + a_nu * dot16(&nu_profiles_2blade[i], &psi_nu_j));
                        m_ch.write(i, j, m_ch.read(i, j) + a_ch * dot16(&ch_profiles_2blade[i], &psi_ch_j));
                    }
                }

                let m_ch_s = (&m_ch + m_ch.transpose()) * faer::scale(0.5);
                let m_nu_s = (&m_nu + m_nu.transpose()) * faer::scale(0.5);
                let eig_ch = m_ch_s.selfadjoint_eigendecomposition(faer::Side::Lower);
                let eig_nu = m_nu_s.selfadjoint_eigendecomposition(faer::Side::Lower);
                let u_raw = eig_ch.u().transpose() * eig_nu.u();
                let (u_pmns, _, _) = crate::quark_sector::extract_ckm_permutation_aware(&u_raw);
                let (t12, t13, t23) = extract_pmns_angles(&u_pmns);

                let mut ev: Vec<f64> = (0..3).map(|i| eig_nu.s().column_vector().read(i).abs()).collect();
                ev.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let dm21 = ev[1] * ev[1] - ev[0] * ev[0];
                let dm31 = ev[2] * ev[2] - ev[0] * ev[0];
                let r = if dm31.abs() > 1e-30 { dm21 / dm31 } else { f64::MAX };

                let chi2 = ((t12 - pdg.theta_12_deg) / pdg.theta_12_err).powi(2)
                         + ((t13 - pdg.theta_13_deg) / pdg.theta_13_err).powi(2)
                         + ((t23 - pdg.theta_23_deg) / pdg.theta_23_err).powi(2);
                let r_pen = ((r - pdg_r) / pdg_r).powi(2) * 100.0;
                let score = chi2 + r_pen;

                if score < best_score {
                    best_score = score;
                    best_result = (a_ch, a_nu, t12, t13, t23, r, chi2);
                }
            }
        }

        let (a_ch, a_nu, t12, t13, t23, r, chi2) = best_result;

        // === Scaled version: dampen diagonal to balance with off-diagonal ===
        println!("\n  --- Scaled diagonal scan (damping factor beta) ---");

        let mut best_scaled = (f64::MAX, 0.0_f64, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        for beta_step in 1..=20 {
            let beta = beta_step as f64 * 0.05; // 0.05 to 1.0
            for ach_step in 0..=10 {
                for anu_step in 0..=10 {
                    let a_ch_s = ach_step as f64;
                    let a_nu_s = anu_step as f64 * 0.3;

                    let cb = construct_casimir_baseline(crate::quark_sector::SubalgebraScheme::InterleavedStride);
                    let (m_base_ch, m_base_nu) = assemble_lepton_baseline(&cb);
                    let mut m_ch_s = m_base_ch;
                    let mut m_nu_s = m_base_nu;

                    // Diagonal: SCALED 3-blade friction
                    for g in 0..3 {
                        let f_ch = beta * (w1 * sel_ch_3blade[g] + w2 * sel_nu_3blade[g]);
                        let f_nu = beta * (w1 * sel_nu_3blade[g] + w2 * sel_ch_3blade[g]);
                        m_ch_s.write(g, g, m_ch_s.read(g, g) + f_ch.exp());
                        m_nu_s.write(g, g, m_nu_s.read(g, g) + f_nu.exp());
                    }

                    // Off-diagonal: 2-blade psi coupling (unscaled)
                    for i in 0..3 {
                        for j in 0..3 {
                            if i == j { continue; }
                            let psi_nu_j = gourlay_psi(&nu_profiles_2blade[j]);
                            let psi_ch_j = gourlay_psi(&ch_profiles_2blade[j]);
                            m_nu_s.write(i, j, m_nu_s.read(i, j) + a_nu_s * dot16(&nu_profiles_2blade[i], &psi_nu_j));
                            m_ch_s.write(i, j, m_ch_s.read(i, j) + a_ch_s * dot16(&ch_profiles_2blade[i], &psi_ch_j));
                        }
                    }

                    let m_ch_sym = (&m_ch_s + m_ch_s.transpose()) * faer::scale(0.5);
                    let m_nu_sym = (&m_nu_s + m_nu_s.transpose()) * faer::scale(0.5);
                    let eig_ch = m_ch_sym.selfadjoint_eigendecomposition(faer::Side::Lower);
                    let eig_nu = m_nu_sym.selfadjoint_eigendecomposition(faer::Side::Lower);
                    let u_raw = eig_ch.u().transpose() * eig_nu.u();
                    let (u_pmns, _, _) = crate::quark_sector::extract_ckm_permutation_aware(&u_raw);
                    let (t12_s, t13_s, t23_s) = extract_pmns_angles(&u_pmns);

                    let mut ev: Vec<f64> = (0..3).map(|i| eig_nu.s().column_vector().read(i).abs()).collect();
                    ev.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let dm21 = ev[1] * ev[1] - ev[0] * ev[0];
                    let dm31 = ev[2] * ev[2] - ev[0] * ev[0];
                    let r_s = if dm31.abs() > 1e-30 { dm21 / dm31 } else { f64::MAX };

                    let chi2_s = ((t12_s - pdg.theta_12_deg) / pdg.theta_12_err).powi(2)
                               + ((t13_s - pdg.theta_13_deg) / pdg.theta_13_err).powi(2)
                               + ((t23_s - pdg.theta_23_deg) / pdg.theta_23_err).powi(2);
                    let r_pen = ((r_s - pdg_r) / pdg_r).powi(2) * 100.0;
                    let score = chi2_s + r_pen;

                    if score < best_scaled.0 {
                        best_scaled = (score, beta, a_ch_s, a_nu_s, t12_s, t13_s, t23_s, r_s);
                    }
                }
            }
        }

        println!("\n  === BEST SCALED TWO-SELECTOR RESULT ===");
        println!("  beta = {:.2} (diagonal damping)", best_scaled.1);
        println!("  alpha_ch = {:.1}, alpha_nu = {:.1}", best_scaled.2, best_scaled.3);
        println!("  theta_12 = {:.2} deg (PDG: {:.2}, err: {:.1}%)", best_scaled.4, pdg.theta_12_deg, ((best_scaled.4 - pdg.theta_12_deg) / pdg.theta_12_deg * 100.0).abs());
        println!("  theta_13 = {:.2} deg (PDG: {:.2}, err: {:.1}%)", best_scaled.5, pdg.theta_13_deg, ((best_scaled.5 - pdg.theta_13_deg) / pdg.theta_13_deg * 100.0).abs());
        println!("  theta_23 = {:.2} deg (PDG: {:.2}, err: {:.1}%)", best_scaled.6, pdg.theta_23_deg, ((best_scaled.6 - pdg.theta_23_deg) / pdg.theta_23_deg * 100.0).abs());
        println!("  r = {:.4} (PDG: {:.4}, err: {:.1}%)", best_scaled.7, pdg_r, ((best_scaled.7 - pdg_r) / pdg_r * 100.0).abs());

        println!("\n  === ORIGINAL (UNSCALED) RESULT ===");
        println!("  alpha_ch = {:.1}, alpha_nu = {:.1}", a_ch, a_nu);
        println!("  theta_12 = {:.2} deg (PDG: {:.2}, err: {:.1}%)", t12, pdg.theta_12_deg, ((t12 - pdg.theta_12_deg) / pdg.theta_12_deg * 100.0).abs());
        println!("  theta_13 = {:.2} deg (PDG: {:.2}, err: {:.1}%)", t13, pdg.theta_13_deg, ((t13 - pdg.theta_13_deg) / pdg.theta_13_deg * 100.0).abs());
        println!("  theta_23 = {:.2} deg (PDG: {:.2}, err: {:.1}%)", t23, pdg.theta_23_deg, ((t23 - pdg.theta_23_deg) / pdg.theta_23_deg * 100.0).abs());
        println!("  r = {:.4} (PDG: {:.4}, err: {:.1}%)", r, pdg_r, ((r - pdg_r) / pdg_r * 100.0).abs());
        println!("  chi2_angles = {:.1}", chi2);
        println!("  score = {:.1}", best_score);
    }

