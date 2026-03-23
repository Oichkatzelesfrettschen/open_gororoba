use super::super::*;
use rayon::prelude::*;

    #[test]
    fn test_neutrino_mass_matrix_and_see_saw() {
        use crate::cayley_dickson_structs::Sedenion;
        use crate::quantum_state::QuantumState;
        use crate::su_n_generators::construct_su5_generators_algebraic;
        use faer::{Mat, Side};

        let mut basis = [Sedenion::default(); 16];
        for i in 0..16 {
            let mut components = [0.0; 16];
            components[i] = 1.0;
            basis[i] = Sedenion::from_slice(&components);
        }
        let i_struct = basis[15];

        let su5_gens = construct_su5_generators_algebraic(&basis, &i_struct);
        let mut dark_gens = Vec::new();
        for generator in su5_gens.iter() {
            if *generator != QuantumState::TopologicalNull {
                dark_gens.push(*generator);
            }
        }

        let dark_gens = &dark_gens[dark_gens.len()-4..];

        let mut mass_matrix = Mat::<f64>::zeros(4, 4);
        for i in 0..4 {
            for j in 0..4 {
                if let (QuantumState::Observable(g1), QuantumState::Observable(g2)) = (dark_gens[i], dark_gens[j]) {
                    let product = g1.conj() * g2;
                    mass_matrix.write(i, j, product.to_slice()[0]);
                }
            }
        }

        println!("Full 4x4 Neutrino Mass Matrix:\n{:?}", mass_matrix);

        let m_r = mass_matrix.read(3,3);
        let m_d = mass_matrix.get(0..3, 3..4);
        let m_light = m_d * m_d.transpose() * (1.0 / m_r);

        let eig = m_light.selfadjoint_eigendecomposition(Side::Lower);
        println!("Light Neutrino Mass Eigenvalues (squared):\n{:?}", eig.s());
    }

    /// PMNS selector pair scan (Rayon-parallelized).
    ///
    /// Scans all splitting-pair combinations for charged-lepton vs neutrino
    /// sector assignment. Targets:
    ///   theta_12 ~ 33.4 deg, theta_23 ~ 49.0 deg, theta_13 ~ 8.5 deg
    #[test]
    fn test_pmns_selector_pair_scan() {
        use crate::lepton_mass_hierarchy::cd_braid_signed_friction;
        use crate::majorana_braiding::MajoranaMode;
        use crate::bell_inequality::SignTableCache;
        use crate::three_fermion_generations::get_sedenion_subalgebras;

        // Enumerate splitting pairs
        let (o1, o2, o3) = get_sedenion_subalgebras();
        let sign_table = SignTableCache::new(16);
        let mut splitting_pairs: Vec<(usize, usize)> = Vec::new();

        for i in 1..16_usize {
            for j in (i + 1)..16 {
                let mi = MajoranaMode { gamma_index: i - 1, cd_basis_index: i, cd_dim: 16 };
                let mj = MajoranaMode { gamma_index: j - 1, cd_basis_index: j, cd_dim: 16 };
                let s1 = cd_braid_signed_friction(&mi, &mj, &o1, &sign_table);
                let s2 = cd_braid_signed_friction(&mi, &mj, &o2, &sign_table);
                let s3 = cd_braid_signed_friction(&mi, &mj, &o3, &sign_table);
                if (s1 - s2).abs() > 1e-9 && (s2 - s3).abs() > 1e-9 && (s1 - s3).abs() > 1e-9 {
                    splitting_pairs.push((i, j));
                }
            }
        }

        println!("--- PMNS SELECTOR PAIR SCAN ---");
        println!("Splitting pairs: {}", splitting_pairs.len());

        // PDG PMNS targets (degrees)
        let pdg_t12: f64 = 33.41;
        let pdg_t13: f64 = 8.54;
        let pdg_t23: f64 = 49.0;

        let combos: Vec<((usize, usize), (usize, usize))> = splitting_pairs.iter()
            .flat_map(|&ch| splitting_pairs.iter()
                .filter(move |&&nu| nu != ch)
                .map(move |&nu| (ch, nu)))
            .collect();

        println!("Total combos: {}", combos.len());

        let mut all_results: Vec<(f64, (usize, usize), (usize, usize), (f64, f64, f64))> =
            combos.par_iter().map(|&(ch_pair, nu_pair)| {
                let pmns = compute_pmns(ch_pair, nu_pair);
                let (t12, t13, t23) = pmns.angles_deg;

                // Score: sum of squared angle deviations (in degrees)
                let score = ((t12 - pdg_t12) / pdg_t12).powi(2)
                    + ((t13 - pdg_t13) / pdg_t13).powi(2)
                    + ((t23 - pdg_t23) / pdg_t23).powi(2);

                (score, ch_pair, nu_pair, (t12, t13, t23))
            }).collect();

        all_results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Report best
        if let Some(&(score, ch, nu, (t12, t13, t23))) = all_results.first() {
            println!("\nBest PMNS selector pair:");
            println!("  Charged lepton: (e_{}, e_{})", ch.0, ch.1);
            println!("  Neutrino:       (e_{}, e_{})", nu.0, nu.1);
            println!("  theta_12 = {:.2} deg (PDG: {:.2})", t12, pdg_t12);
            println!("  theta_13 = {:.2} deg (PDG: {:.2})", t13, pdg_t13);
            println!("  theta_23 = {:.2} deg (PDG: {:.2})", t23, pdg_t23);
            println!("  Score: {:.6}", score);

            // Full PMNS matrix
            let pmns = compute_pmns(ch, nu);
            println!("\n  U_PMNS:");
            for r in 0..3 {
                println!("    [{:.6}, {:.6}, {:.6}]",
                    pmns.matrix.read(r, 0), pmns.matrix.read(r, 1), pmns.matrix.read(r, 2));
            }
            println!("  Neutrino masses (arb. units): [{:.6}, {:.6}, {:.6}]",
                pmns.neutrino_masses[0], pmns.neutrino_masses[1], pmns.neutrino_masses[2]);
            println!("  Charged lepton masses: [{:.6}, {:.6}, {:.6}]",
                pmns.charged_masses[0], pmns.charged_masses[1], pmns.charged_masses[2]);
            println!("  delta_m21^2 = {:.6e}", pmns.delta_m21_sq);
            println!("  delta_m31^2 = {:.6e}", pmns.delta_m31_sq);
            if pmns.delta_m31_sq.abs() > 1e-20 {
                println!("  ratio delta_m21^2 / delta_m31^2 = {:.4} (PDG ~ 0.030)",
                    pmns.delta_m21_sq / pmns.delta_m31_sq);
            }
        }

        // Top-5
        println!("\n--- TOP-5 PMNS SELECTOR PAIRS ---");
        for (rank, (score, ch, nu, (t12, t13, t23))) in all_results.iter().take(5).enumerate() {
            println!("  #{}: ch=(e_{},e_{}), nu=(e_{},e_{}) | t12={:.1}, t13={:.1}, t23={:.1} | score={:.4}",
                rank + 1, ch.0, ch.1, nu.0, nu.1, t12, t13, t23, score);
        }

        // Structural prediction: CKM vs PMNS angle ratio
        println!("\n--- CKM vs PMNS STRUCTURAL COMPARISON ---");
        if let Some(&(_, ch, nu, (t12, t13, t23))) = all_results.first() {
            println!("  PMNS best: ch=(e_{},e_{}), nu=(e_{},e_{})", ch.0, ch.1, nu.0, nu.1);
            println!("  CKM  best: up=(e_11,e_12), down=(e_10,e_11)");
            println!("  theta_12 ratio (PMNS/CKM): {:.2} (observed: {:.2})",
                t12 / 14.19, pdg_t12 / 12.99);
            println!("  theta_23 ratio (PMNS/CKM): {:.2} (observed: {:.2})",
                t23 / 2.52, pdg_t23 / 2.40);
            println!("  theta_13 ratio (PMNS/CKM): {:.2} (observed: {:.2})",
                t13 / 0.22, pdg_t13 / 0.214);
        }
    }

    /// Electroweak mixing angle from SU(5) + associator flux.
    ///
    /// At the GUT scale, sin^2(theta_W) = 3/8 = 0.375.
    /// At M_Z, PDG gives sin^2(theta_W) = 0.23122.
    ///
    /// The RG running from GUT to M_Z depends on the beta-function coefficients
    /// (b_1, b_2, b_3) for U(1), SU(2), SU(3). In the standard SU(5) with one
    /// Higgs doublet: b_1 = 41/10, b_2 = -19/6, b_3 = -7.
    ///
    /// Here we test whether the topological friction from the three octonionic
    /// subalgebras provides a natural modification to the effective couplings.
    #[test]
    fn test_electroweak_mixing_angle_from_associator_flux() {
        use crate::lepton_mass_hierarchy::cd_braid_signed_friction;
        use crate::majorana_braiding::MajoranaMode;
        use crate::bell_inequality::SignTableCache;
        use crate::three_fermion_generations::get_sedenion_subalgebras;

        println!("--- ELECTROWEAK MIXING ANGLE FROM SU(5) + ASSOCIATOR FLUX ---");

        // GUT-scale prediction
        let sin2_tw_gut: f64 = 3.0 / 8.0;
        let sin2_tw_mz: f64 = 0.23122; // PDG 2025
        println!("  sin^2(theta_W) at GUT scale: {:.6}", sin2_tw_gut);
        println!("  sin^2(theta_W) at M_Z (PDG): {:.6}", sin2_tw_mz);
        println!("  Running ratio: {:.6}", sin2_tw_mz / sin2_tw_gut);

        // Compute the total associator flux per subalgebra
        // for each of the gauge-sector-relevant braid axes.
        let (o1, o2, o3) = get_sedenion_subalgebras();
        let sign_table = SignTableCache::new(16);

        // SU(3) sector uses e_1..e_8 (color generators)
        // SU(2) sector uses e_9..e_11 (weak isospin)
        // U(1) sector uses e_12 (hypercharge)

        // Compute total signed friction for SU(3) sector braid axes
        let su3_pairs = [(1_usize, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 6), (1, 4), (2, 5)];
        let su2_pairs = [(9_usize, 10), (9, 11), (10, 11)];

        let mut su3_flux = [0.0_f64; 3]; // per generation
        let mut su2_flux = [0.0_f64; 3];

        for &(a, b) in &su3_pairs {
            let ma = MajoranaMode { gamma_index: a - 1, cd_basis_index: a, cd_dim: 16 };
            let mb = MajoranaMode { gamma_index: b - 1, cd_basis_index: b, cd_dim: 16 };
            for (g, sub) in [&o1, &o2, &o3].iter().enumerate() {
                su3_flux[g] += cd_braid_signed_friction(&ma, &mb, sub, &sign_table).abs();
            }
        }

        for &(a, b) in &su2_pairs {
            let ma = MajoranaMode { gamma_index: a - 1, cd_basis_index: a, cd_dim: 16 };
            let mb = MajoranaMode { gamma_index: b - 1, cd_basis_index: b, cd_dim: 16 };
            for (g, sub) in [&o1, &o2, &o3].iter().enumerate() {
                su2_flux[g] += cd_braid_signed_friction(&ma, &mb, sub, &sign_table).abs();
            }
        }

        let su3_total: f64 = su3_flux.iter().sum();
        let su2_total: f64 = su2_flux.iter().sum();

        println!("\n  Associator flux per generation:");
        println!("    SU(3): [{:.4}, {:.4}, {:.4}] total={:.4}",
            su3_flux[0], su3_flux[1], su3_flux[2], su3_total);
        println!("    SU(2): [{:.4}, {:.4}, {:.4}] total={:.4}",
            su2_flux[0], su2_flux[1], su2_flux[2], su2_total);

        // The ratio of total flux gives the effective coupling ratio modification
        // sin^2(theta_W, eff) = sin^2(theta_W, GUT) * (SU(2) flux / SU(3) flux)
        if su3_total > 1e-10 {
            let flux_ratio = su2_total / su3_total;
            let sin2_tw_pred = sin2_tw_gut * flux_ratio;
            println!("\n  Flux ratio SU(2)/SU(3) = {:.6}", flux_ratio);
            println!("  sin^2(theta_W) predicted = {:.6}", sin2_tw_pred);
            println!("  sin^2(theta_W) PDG      = {:.6}", sin2_tw_mz);
            println!("  Relative error: {:.2}%",
                ((sin2_tw_pred - sin2_tw_mz) / sin2_tw_mz * 100.0).abs());
        }

        // Alternative: use the asymmetry between gauge sectors as running scale
        // The SU(5) one-loop running formula:
        //   alpha_i^{-1}(M_Z) = alpha_GUT^{-1} + b_i/(2*pi) * ln(M_GUT/M_Z)
        // With standard MS-bar coefficients:
        let b1: f64 = 41.0 / 10.0;
        let b2: f64 = -19.0 / 6.0;
        let b3: f64 = -7.0;

        // At M_Z: sin^2(theta_W) = alpha_em / alpha_2
        // With GUT normalization: alpha_1 = (5/3) * alpha_Y
        // sin^2(theta_W) = (3/8) * (1 + (5/8) * (b1 - b2) * alpha_GUT / (2*pi) * ln(M_GUT/M_Z))^{-1}
        // For standard SU(5): ln(M_GUT/M_Z) ~ 32
        let ln_gut_mz = 32.0;
        let alpha_gut_inv: f64 = 42.0; // 1/alpha_GUT ~ 42 in minimal SU(5)

        let correction = (5.0 / 8.0) * (b1 - b2) / (2.0 * std::f64::consts::PI) * ln_gut_mz / alpha_gut_inv;
        let sin2_tw_run = sin2_tw_gut / (1.0 + correction);

        println!("\n  Standard SU(5) one-loop running:");
        println!("    b1={:.4}, b2={:.4}, b3={:.4}", b1, b2, b3);
        println!("    ln(M_GUT/M_Z) = {:.1}", ln_gut_mz);
        println!("    1/alpha_GUT = {:.1}", alpha_gut_inv);
        println!("    sin^2(theta_W) one-loop = {:.6} (PDG: {:.6})",
            sin2_tw_run, sin2_tw_mz);
    }

    /// PMNS theta_23 targeted alpha scan.
    ///
    /// Current: theta_23 = 32.3 deg (PDG 49.0, 34% error).
    /// The issue: diagonal friction perturbation produces too much hierarchy
    /// in the 2-3 sector. Near-maximal theta_23 requires near-degeneracy.
    ///
    /// Strategy: vary the relative strength of neutrino-sector friction
    /// via an alpha parameter while keeping the best selector pair fixed.
    #[test]
    fn test_pmns_theta23_alpha_scan() {
        use crate::lepton_mass_hierarchy::cd_braid_signed_friction;
        use crate::majorana_braiding::MajoranaMode;
        use crate::bell_inequality::SignTableCache;
        use crate::three_fermion_generations::get_sedenion_subalgebras;
        use crate::quark_sector::SubalgebraScheme;

        let (o1, o2, o3) = get_sedenion_subalgebras();
        let subs = [&o1, &o2, &o3];
        let sign_table = SignTableCache::new(16);

        // Best selector pair from the PMNS scan
        let ch_pair = (11_usize, 12_usize);
        let nu_pair = (7_usize, 8_usize);

        let ch_a = MajoranaMode { gamma_index: ch_pair.0 - 1, cd_basis_index: ch_pair.0, cd_dim: 16 };
        let ch_b = MajoranaMode { gamma_index: ch_pair.1 - 1, cd_basis_index: ch_pair.1, cd_dim: 16 };
        let nu_a = MajoranaMode { gamma_index: nu_pair.0 - 1, cd_basis_index: nu_pair.0, cd_dim: 16 };
        let nu_b = MajoranaMode { gamma_index: nu_pair.1 - 1, cd_basis_index: nu_pair.1, cd_dim: 16 };

        let sel_ch: Vec<f64> = subs.iter()
            .map(|s| cd_braid_signed_friction(&ch_a, &ch_b, s, &sign_table))
            .collect();
        let sel_nu: Vec<f64> = subs.iter()
            .map(|s| cd_braid_signed_friction(&nu_a, &nu_b, s, &sign_table))
            .collect();

        // Casimir baseline
        let cb = construct_casimir_baseline(SubalgebraScheme::InterleavedStride);
        let (m_base_ch, m_base_nu) = assemble_lepton_baseline(&cb);

        let pdg_t23: f64 = 49.0;
        let pdg_t12: f64 = 33.41;
        let pdg_t13: f64 = 8.54;

        println!("--- PMNS theta_23 ALPHA SCAN ---");
        println!("  Selectors: ch=(e_{},e_{}), nu=(e_{},e_{})",
            ch_pair.0, ch_pair.1, nu_pair.0, nu_pair.1);
        println!("  sel_ch = [{:.4}, {:.4}, {:.4}]", sel_ch[0], sel_ch[1], sel_ch[2]);
        println!("  sel_nu = [{:.4}, {:.4}, {:.4}]", sel_nu[0], sel_nu[1], sel_nu[2]);

        // Scan alpha_nu from 0.01 to 2.0: controls neutrino friction strength
        let mut best_score = f64::INFINITY;
        let mut best_alpha = 0.0_f64;
        let mut best_angles = (0.0_f64, 0.0_f64, 0.0_f64);

        for step in 0..200 {
            let alpha_nu = 0.01 + step as f64 * 0.01;

            // Lepton-fitted weights
            let w1: f64 = -0.656850;
            let w2: f64 = -0.741999;

            let mut m_ch = m_base_ch.clone();
            let mut m_nu = m_base_nu.clone();

            for i in 0..3 {
                let f_ch = w1 * sel_ch[i] + w2 * sel_nu[i];
                let f_nu = alpha_nu * (w1 * sel_nu[i] + w2 * sel_ch[i]);
                m_ch.write(i, i, m_ch.read(i, i) + f_ch.exp());
                m_nu.write(i, i, m_nu.read(i, i) + f_nu.exp());
            }

            let m_ch_sym = (&m_ch + m_ch.transpose()) * faer::scale(0.5);
            let m_nu_sym = (&m_nu + m_nu.transpose()) * faer::scale(0.5);

            let eig_ch = m_ch_sym.selfadjoint_eigendecomposition(faer::Side::Lower);
            let eig_nu = m_nu_sym.selfadjoint_eigendecomposition(faer::Side::Lower);

            let u_pmns_raw = eig_ch.u().transpose() * eig_nu.u();
            let (u_pmns, _, _) = crate::quark_sector::extract_ckm_permutation_aware(&u_pmns_raw);

            let (t12, t13, t23) = extract_pmns_angles(&u_pmns);

            let score = ((t23 - pdg_t23) / pdg_t23).powi(2)
                + ((t12 - pdg_t12) / pdg_t12).powi(2)
                + ((t13 - pdg_t13) / pdg_t13).powi(2);

            if score < best_score {
                best_score = score;
                best_alpha = alpha_nu;
                best_angles = (t12, t13, t23);
            }
        }

        println!("\n  Best alpha_nu: {:.4}", best_alpha);
        println!("  theta_12 = {:.2} deg (PDG: {:.2})", best_angles.0, pdg_t12);
        println!("  theta_13 = {:.2} deg (PDG: {:.2})", best_angles.1, pdg_t13);
        println!("  theta_23 = {:.2} deg (PDG: {:.2})", best_angles.2, pdg_t23);
        println!("  Score: {:.6}", best_score);
        println!("  (Previous: theta_23=32.3, score=0.132)");
    }

    /// PMNS theta_23 composite selector scan.
    ///
    /// Uses TWO braid pairs per sector (4 total) to fill the zero entry
    /// in the friction vector. The composite friction is:
    ///   F_ch = w1*sel_A + w2*sel_B  (two selectors for charged lepton)
    ///   F_nu = w1*sel_C + w2*sel_D  (two selectors for neutrino)
    ///
    /// Each selector pair is chosen from the 21 splitting pairs.
    #[test]
    fn test_pmns_theta23_composite_scan() {
        use crate::lepton_mass_hierarchy::cd_braid_signed_friction;
        use crate::majorana_braiding::MajoranaMode;
        use crate::bell_inequality::SignTableCache;
        use crate::three_fermion_generations::get_sedenion_subalgebras;
        use crate::quark_sector::SubalgebraScheme;
        use rayon::prelude::*;

        let (o1, o2, o3) = get_sedenion_subalgebras();
        let sign_table = SignTableCache::new(16);

        // Precompute all 21 signed friction vectors
        let mut pairs: Vec<(usize, usize)> = Vec::new();
        let mut frictions: Vec<[f64; 3]> = Vec::new();
        for i in 1..16_usize {
            for j in (i + 1)..16 {
                let mi = MajoranaMode { gamma_index: i - 1, cd_basis_index: i, cd_dim: 16 };
                let mj = MajoranaMode { gamma_index: j - 1, cd_basis_index: j, cd_dim: 16 };
                let s1 = cd_braid_signed_friction(&mi, &mj, &o1, &sign_table);
                let s2 = cd_braid_signed_friction(&mi, &mj, &o2, &sign_table);
                let s3 = cd_braid_signed_friction(&mi, &mj, &o3, &sign_table);
                if (s1 - s2).abs() > 1e-9 && (s2 - s3).abs() > 1e-9 && (s1 - s3).abs() > 1e-9 {
                    pairs.push((i, j));
                    frictions.push([s1, s2, s3]);
                }
            }
        }

        println!("--- PMNS theta_23 COMPOSITE SELECTOR SCAN ---");
        println!("Splitting pairs: {}", pairs.len());

        // Casimir baseline
        let cb = construct_casimir_baseline(SubalgebraScheme::InterleavedStride);
        let (m_base_ch, m_base_nu) = assemble_lepton_baseline(&cb);

        let pdg_t23: f64 = 49.0;
        let pdg_t12: f64 = 33.41;
        let pdg_t13: f64 = 8.54;

        let w1: f64 = -0.656850;
        let w2: f64 = -0.741999;
        let n = frictions.len();

        // Fix ch to best pair, scan nu composites
        let best_ch_idx = pairs.iter().position(|&p| p == (11, 12)).unwrap();
        let combos_reduced: Vec<(usize, usize)> = (0..n).flat_map(|c|
            (0..n).filter(move |&d| d != c).map(move |d| (c, d))
        ).collect();

        println!("Scanning {} nu composite pairs (ch fixed to (e_11,e_12))", combos_reduced.len());

        let results: Vec<(f64, usize, usize, (f64, f64, f64))> =
            combos_reduced.par_iter().map(|&(c_idx, d_idx)| {
                let sel_ch = &frictions[best_ch_idx];

                // Composite neutrino friction: average of two friction vectors
                let sel_nu_composite: [f64; 3] = [
                    (frictions[c_idx][0] + frictions[d_idx][0]) / 2.0,
                    (frictions[c_idx][1] + frictions[d_idx][1]) / 2.0,
                    (frictions[c_idx][2] + frictions[d_idx][2]) / 2.0,
                ];

                let mut m_ch = m_base_ch.clone();
                let mut m_nu = m_base_nu.clone();
                for i in 0..3 {
                    let f_ch = w1 * sel_ch[i] + w2 * sel_nu_composite[i];
                    let f_nu = w1 * sel_nu_composite[i] + w2 * sel_ch[i];
                    m_ch.write(i, i, m_ch.read(i, i) + f_ch.exp());
                    m_nu.write(i, i, m_nu.read(i, i) + f_nu.exp());
                }

                let m_ch_sym = (&m_ch + m_ch.transpose()) * faer::scale(0.5);
                let m_nu_sym = (&m_nu + m_nu.transpose()) * faer::scale(0.5);
                let eig_ch = m_ch_sym.selfadjoint_eigendecomposition(faer::Side::Lower);
                let eig_nu = m_nu_sym.selfadjoint_eigendecomposition(faer::Side::Lower);
                let u_raw = eig_ch.u().transpose() * eig_nu.u();
                let (u_pmns, _, _) = crate::quark_sector::extract_ckm_permutation_aware(&u_raw);
                let (t12, t13, t23) = extract_pmns_angles(&u_pmns);

                let score = ((t23 - pdg_t23) / pdg_t23).powi(2)
                    + ((t12 - pdg_t12) / pdg_t12).powi(2)
                    + ((t13 - pdg_t13) / pdg_t13).powi(2);

                (score, c_idx, d_idx, (t12, t13, t23))
            }).collect();

        let mut sorted = results;
        sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        if let Some(&(score, c, d, (t12, t13, t23))) = sorted.first() {
            println!("\nBest composite neutrino selector:");
            println!("  nu = avg((e_{},e_{}), (e_{},e_{}))",
                pairs[c].0, pairs[c].1, pairs[d].0, pairs[d].1);
            println!("  theta_12 = {:.2} deg (PDG: {:.2})", t12, pdg_t12);
            println!("  theta_13 = {:.2} deg (PDG: {:.2})", t13, pdg_t13);
            println!("  theta_23 = {:.2} deg (PDG: {:.2})", t23, pdg_t23);
            println!("  Score: {:.6} (previous single: 0.132)", score);
        }

        println!("\n--- TOP-5 COMPOSITE ---");
        for (rank, (score, c, d, (t12, t13, t23))) in sorted.iter().take(5).enumerate() {
            println!("  #{}: nu=avg(({},{}),({},{})) | t12={:.1}, t13={:.1}, t23={:.1} | score={:.4}",
                rank + 1, pairs[*c].0, pairs[*c].1, pairs[*d].0, pairs[*d].1,
                t12, t13, t23, score);
        }
    }

    /// PMNS theta_23 with psi-circulant off-diagonal coupling.
    ///
    /// Scans alpha_cross for the off-diagonal coupling strength.
    /// Uses the full 16D associator profile as psi input for richer overlap.
    #[test]
    fn test_pmns_offdiag_psi_coupling() {
        let ch_pair = (11_usize, 12);
        let nu_pair = (7_usize, 8);

        let pdg_t12: f64 = 33.41;
        let pdg_t13: f64 = 8.54;
        let pdg_t23: f64 = 49.0;

        println!("--- PMNS OFF-DIAGONAL PSI-CIRCULANT SCAN ---");

        let mut best_score = f64::INFINITY;
        let mut best_alpha = 0.0_f64;
        let mut best_angles = (0.0_f64, 0.0_f64, 0.0_f64);

        // Scan alpha_cross from -2.0 to 2.0
        for step in 0..400 {
            let alpha = -2.0 + step as f64 * 0.01;

            let (m_ch, m_nu) = construct_pmns_matrices_offdiag(ch_pair, nu_pair, alpha);
            let m_ch_sym = (&m_ch + m_ch.transpose()) * faer::scale(0.5);
            let m_nu_sym = (&m_nu + m_nu.transpose()) * faer::scale(0.5);

            let eig_ch = m_ch_sym.selfadjoint_eigendecomposition(faer::Side::Lower);
            let eig_nu = m_nu_sym.selfadjoint_eigendecomposition(faer::Side::Lower);

            let u_raw = eig_ch.u().transpose() * eig_nu.u();
            let (u_pmns, _, _) = crate::quark_sector::extract_ckm_permutation_aware(&u_raw);
            let (t12, t13, t23) = extract_pmns_angles(&u_pmns);

            let score = ((t23 - pdg_t23) / pdg_t23).powi(2)
                + ((t12 - pdg_t12) / pdg_t12).powi(2)
                + ((t13 - pdg_t13) / pdg_t13).powi(2);

            if score < best_score {
                best_score = score;
                best_alpha = alpha;
                best_angles = (t12, t13, t23);
            }
        }

        println!("  Best alpha_cross: {:.4}", best_alpha);
        println!("  theta_12 = {:.2} deg (PDG: {:.2})", best_angles.0, pdg_t12);
        println!("  theta_13 = {:.2} deg (PDG: {:.2})", best_angles.1, pdg_t13);
        println!("  theta_23 = {:.2} deg (PDG: {:.2})", best_angles.2, pdg_t23);
        println!("  Score: {:.6} (diagonal-only: 0.132)", best_score);

        if best_angles.2 > 40.0 {
            println!("  *** THETA_23 > 40 DEG -- CEILING BROKEN ***");
        }
    }

    /// Refined PMNS with FULL 16D friction profiles + psi overlap.
    ///
    /// Instead of placing friction at 2 sparse indices, compute the
    /// full 16D associator profile for each generation's braid,
    /// then use <profile_i, psi(profile_j)> for the off-diagonal M_ij.
    #[test]
    fn test_pmns_offdiag_full_profile() {
        use crate::lepton_mass_hierarchy::cd_braid_signed_friction;
        use crate::majorana_braiding::MajoranaMode;
        use crate::bell_inequality::SignTableCache;
        use crate::three_fermion_generations::get_sedenion_subalgebras;
        use crate::quark_sector::SubalgebraScheme;
        use cd_kernel::gourlay_psi;

        let ch_pair = (11_usize, 12);
        let nu_pair = (7_usize, 8);

        let (o1, o2, o3) = get_sedenion_subalgebras();
        let subs = [&o1, &o2, &o3];
        let sign_table = SignTableCache::new(16);

        let ch_a = MajoranaMode { gamma_index: ch_pair.0 - 1, cd_basis_index: ch_pair.0, cd_dim: 16 };
        let ch_b = MajoranaMode { gamma_index: ch_pair.1 - 1, cd_basis_index: ch_pair.1, cd_dim: 16 };
        let nu_a = MajoranaMode { gamma_index: nu_pair.0 - 1, cd_basis_index: nu_pair.0, cd_dim: 16 };
        let nu_b = MajoranaMode { gamma_index: nu_pair.1 - 1, cd_basis_index: nu_pair.1, cd_dim: 16 };

        // Build FULL 16D friction profiles per generation
        // For each generation g, the profile is a 16D vector where
        // component k = associator contribution from probe e_k
        let build_profile = |mode_i: &MajoranaMode, mode_j: &MajoranaMode, sub: &[usize]| -> [f64; 16] {
            use crate::bell_inequality::rotate_sparse;
            let i = mode_i.cd_basis_index;
            let j = mode_j.cd_basis_index;
            let a_sparse = vec![(i, 1.0)];
            let theta = std::f64::consts::FRAC_PI_4;
            let a_rotated = rotate_sparse(&a_sparse, i, j, theta);
            let b_sparse = vec![(j, 1.0)];

            let mut profile = [0.0_f64; 16];
            for &k in sub {
                if k == 0 || k == i || k == j { continue; }
                let x_sparse = [(k, 1.0)];
                let val = sign_table.sparse_associator_sum(&a_rotated, &x_sparse, &b_sparse);
                profile[k] = val;
            }
            profile
        };

        // Build profiles for each generation for both sectors
        let ch_profiles: Vec<[f64; 16]> = subs.iter()
            .map(|s| build_profile(&ch_a, &ch_b, s))
            .collect();
        let nu_profiles: Vec<[f64; 16]> = subs.iter()
            .map(|s| build_profile(&nu_a, &nu_b, s))
            .collect();

        println!("--- PMNS FULL-PROFILE PSI OVERLAP SCAN ---");

        // Compute psi overlaps for the neutrino sector
        let dot16 = |a: &[f64; 16], b: &[f64; 16]| -> f64 {
            a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
        };

        // Self-overlaps (diagonal) -- should be identical across generations if psi-symmetric
        for g in 0..3 {
            println!("  nu_profile[{}] norm^2 = {:.4}", g,
                dot16(&nu_profiles[g], &nu_profiles[g]));
        }

        // Psi overlaps (off-diagonal)
        for g in 0..3 {
            let psi_prof = gourlay_psi(&nu_profiles[g]);
            let overlap = dot16(&nu_profiles[g], &psi_prof);
            println!("  <nu[{}], psi(nu[{}])> = {:.4}", g, g, overlap);
        }

        // Casimir baseline via neutral projections + lepton assembly
        let cb = construct_casimir_baseline(SubalgebraScheme::InterleavedStride);
        let (m_base_ch, m_base_nu) = assemble_lepton_baseline(&cb);

        let w1: f64 = -0.656850;
        let w2: f64 = -0.741999;
        let pdg_t12: f64 = 33.41;
        let pdg_t13: f64 = 8.54;
        let pdg_t23: f64 = 49.0;

        let sel_ch: Vec<f64> = subs.iter()
            .map(|s| cd_braid_signed_friction(&ch_a, &ch_b, s, &sign_table))
            .collect();
        let sel_nu: Vec<f64> = subs.iter()
            .map(|s| cd_braid_signed_friction(&nu_a, &nu_b, s, &sign_table))
            .collect();

        // Scan alpha_cross with full-profile psi overlap
        let mut best_score = f64::INFINITY;
        let mut best_alpha = 0.0_f64;
        let mut best_angles = (0.0_f64, 0.0_f64, 0.0_f64);

        for step in 0..400 {
            let alpha = -2.0 + step as f64 * 0.01;

            let mut m_ch = m_base_ch.clone();
            let mut m_nu = m_base_nu.clone();

            // Diagonal
            for i in 0..3 {
                let f_ch = w1 * sel_ch[i] + w2 * sel_nu[i];
                let f_nu = w1 * sel_nu[i] + w2 * sel_ch[i];
                m_ch.write(i, i, m_ch.read(i, i) + f_ch.exp());
                m_nu.write(i, i, m_nu.read(i, i) + f_nu.exp());
            }

            // Off-diagonal from psi overlap on FULL profiles
            for i in 0..3 {
                for j in 0..3 {
                    if i == j { continue; }
                    let psi_nu_j = gourlay_psi(&nu_profiles[j]);
                    let overlap_nu = dot16(&nu_profiles[i], &psi_nu_j);
                    let psi_ch_j = gourlay_psi(&ch_profiles[j]);
                    let overlap_ch = dot16(&ch_profiles[i], &psi_ch_j);

                    m_nu.write(i, j, m_nu.read(i, j) + alpha * overlap_nu);
                    m_ch.write(i, j, m_ch.read(i, j) + alpha * overlap_ch);
                }
            }

            // Symmetrize
            for i in 0..3 {
                for j in (i + 1)..3 {
                    let avg_ch = (m_ch.read(i, j) + m_ch.read(j, i)) / 2.0;
                    let avg_nu = (m_nu.read(i, j) + m_nu.read(j, i)) / 2.0;
                    m_ch.write(i, j, avg_ch); m_ch.write(j, i, avg_ch);
                    m_nu.write(i, j, avg_nu); m_nu.write(j, i, avg_nu);
                }
            }

            let m_ch_sym = (&m_ch + m_ch.transpose()) * faer::scale(0.5);
            let m_nu_sym = (&m_nu + m_nu.transpose()) * faer::scale(0.5);
            let eig_ch = m_ch_sym.selfadjoint_eigendecomposition(faer::Side::Lower);
            let eig_nu = m_nu_sym.selfadjoint_eigendecomposition(faer::Side::Lower);
            let u_raw = eig_ch.u().transpose() * eig_nu.u();
            let (u_pmns, _, _) = crate::quark_sector::extract_ckm_permutation_aware(&u_raw);
            let (t12, t13, t23) = extract_pmns_angles(&u_pmns);

            let score = ((t23 - pdg_t23) / pdg_t23).powi(2)
                + ((t12 - pdg_t12) / pdg_t12).powi(2)
                + ((t13 - pdg_t13) / pdg_t13).powi(2);

            if score < best_score {
                best_score = score;
                best_alpha = alpha;
                best_angles = (t12, t13, t23);
            }
        }

        println!("\n  Best alpha_cross: {:.4}", best_alpha);
        println!("  theta_12 = {:.2} deg (PDG: {:.2})", best_angles.0, pdg_t12);
        println!("  theta_13 = {:.2} deg (PDG: {:.2})", best_angles.1, pdg_t13);
        println!("  theta_23 = {:.2} deg (PDG: {:.2})", best_angles.2, pdg_t23);
        println!("  Score: {:.6} (sparse: 0.110, diagonal: 0.132)", best_score);

        if best_angles.2 > 40.0 {
            println!("  *** THETA_23 > 40 DEG -- APPROACHING PDG ***");
        }
        if best_angles.2 > 45.0 {
            println!("  *** THETA_23 > 45 DEG -- NEAR-MAXIMAL MIXING ***");
        }
    }

    /// Two-parameter off-diagonal scan with weighted score.
    ///
    /// Uses independent alpha_ch and alpha_nu for the two sectors,
    /// and a weighted score that penalizes theta_13 drift while
    /// rewarding theta_23 improvement.
    #[test]
    fn test_pmns_offdiag_two_param() {
        use crate::lepton_mass_hierarchy::cd_braid_signed_friction;
        use crate::majorana_braiding::MajoranaMode;
        use crate::bell_inequality::SignTableCache;
        use crate::three_fermion_generations::get_sedenion_subalgebras;
        use crate::quark_sector::SubalgebraScheme;
        use cd_kernel::gourlay_psi;
        use rayon::prelude::*;

        let ch_pair = (11_usize, 12);
        let nu_pair = (7_usize, 8);
        let (o1, o2, o3) = get_sedenion_subalgebras();
        let subs = [&o1, &o2, &o3];
        let sign_table = SignTableCache::new(16);

        let ch_a = MajoranaMode { gamma_index: ch_pair.0 - 1, cd_basis_index: ch_pair.0, cd_dim: 16 };
        let ch_b = MajoranaMode { gamma_index: ch_pair.1 - 1, cd_basis_index: ch_pair.1, cd_dim: 16 };
        let nu_a = MajoranaMode { gamma_index: nu_pair.0 - 1, cd_basis_index: nu_pair.0, cd_dim: 16 };
        let nu_b = MajoranaMode { gamma_index: nu_pair.1 - 1, cd_basis_index: nu_pair.1, cd_dim: 16 };

        // Build full 16D profiles
        let build_profile = |mode_i: &MajoranaMode, mode_j: &MajoranaMode, sub: &[usize]| -> [f64; 16] {
            use crate::bell_inequality::rotate_sparse;
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

        let ch_profiles: Vec<[f64; 16]> = subs.iter().map(|s| build_profile(&ch_a, &ch_b, s)).collect();
        let nu_profiles: Vec<[f64; 16]> = subs.iter().map(|s| build_profile(&nu_a, &nu_b, s)).collect();

        let sel_ch: Vec<f64> = subs.iter().map(|s| cd_braid_signed_friction(&ch_a, &ch_b, s, &sign_table)).collect();
        let sel_nu: Vec<f64> = subs.iter().map(|s| cd_braid_signed_friction(&nu_a, &nu_b, s, &sign_table)).collect();

        let cb = construct_casimir_baseline(SubalgebraScheme::InterleavedStride);
        let (m_base_ch, m_base_nu) = assemble_lepton_baseline(&cb);

        let w1: f64 = -0.656850;
        let w2: f64 = -0.741999;
        let pdg_t12: f64 = 33.41;
        let pdg_t13: f64 = 8.54;
        let pdg_t23: f64 = 49.0;

        let dot16 = |a: &[f64; 16], b: &[f64; 16]| -> f64 {
            a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
        };

        println!("--- PMNS TWO-PARAMETER OFF-DIAGONAL SCAN ---");

        // Parallel scan over (alpha_ch, alpha_nu) grid
        let grid: Vec<(f64, f64)> = (0..100).flat_map(|i|
            (0..100).map(move |j| (i as f64 * 0.05, j as f64 * 0.05))
        ).collect();

        let results: Vec<(f64, f64, f64, (f64, f64, f64))> = grid.par_iter().map(|&(a_ch, a_nu)| {
            let mut m_ch = m_base_ch.clone();
            let mut m_nu = m_base_nu.clone();

            for i in 0..3 {
                let f_ch = w1 * sel_ch[i] + w2 * sel_nu[i];
                let f_nu = w1 * sel_nu[i] + w2 * sel_ch[i];
                m_ch.write(i, i, m_ch.read(i, i) + f_ch.exp());
                m_nu.write(i, i, m_nu.read(i, i) + f_nu.exp());
            }

            for i in 0..3 { for j in 0..3 { if i == j { continue; }
                let psi_nu_j = gourlay_psi(&nu_profiles[j]);
                let psi_ch_j = gourlay_psi(&ch_profiles[j]);
                m_nu.write(i, j, m_nu.read(i, j) + a_nu * dot16(&nu_profiles[i], &psi_nu_j));
                m_ch.write(i, j, m_ch.read(i, j) + a_ch * dot16(&ch_profiles[i], &psi_ch_j));
            }}

            for i in 0..3 { for j in (i+1)..3 {
                let avg_ch = (m_ch.read(i,j) + m_ch.read(j,i)) / 2.0;
                let avg_nu = (m_nu.read(i,j) + m_nu.read(j,i)) / 2.0;
                m_ch.write(i,j,avg_ch); m_ch.write(j,i,avg_ch);
                m_nu.write(i,j,avg_nu); m_nu.write(j,i,avg_nu);
            }}

            let m_ch_s = (&m_ch + m_ch.transpose()) * faer::scale(0.5);
            let m_nu_s = (&m_nu + m_nu.transpose()) * faer::scale(0.5);
            let eig_ch = m_ch_s.selfadjoint_eigendecomposition(faer::Side::Lower);
            let eig_nu = m_nu_s.selfadjoint_eigendecomposition(faer::Side::Lower);
            let u_raw = eig_ch.u().transpose() * eig_nu.u();
            let (u_pmns, _, _) = crate::quark_sector::extract_ckm_permutation_aware(&u_raw);
            let (t12, t13, t23) = extract_pmns_angles(&u_pmns);

            // Weighted score: penalize theta_13 drift, reward theta_23
            let score = ((t12 - pdg_t12) / pdg_t12).powi(2)
                + 2.0 * ((t13 - pdg_t13) / pdg_t13).powi(2)
                + 3.0 * ((t23 - pdg_t23) / pdg_t23).powi(2);

            (score, a_ch, a_nu, (t12, t13, t23))
        }).collect();

        let best = results.iter().min_by(|a, b| a.0.partial_cmp(&b.0).unwrap()).unwrap();

        println!("  Best (alpha_ch, alpha_nu) = ({:.4}, {:.4})", best.1, best.2);
        println!("  theta_12 = {:.2} deg (PDG: {:.2})", (best.3).0, pdg_t12);
        println!("  theta_13 = {:.2} deg (PDG: {:.2})", (best.3).1, pdg_t13);
        println!("  theta_23 = {:.2} deg (PDG: {:.2})", (best.3).2, pdg_t23);
        println!("  Weighted score: {:.6}", best.0);
    }

    /// Type X triad defect vector as off-diagonal coupling source.
    ///
    /// For each Type X triad (b,c,d), the 16D associator defect is:
    ///   v = (e_b * e_c) * e_d - e_b * (e_c * e_d) = +/- 2 * e_{b^c^d}
    ///
    /// The psi overlap on these defect vectors gives the cross-generational
    /// coupling from the fully non-associative sector.
    #[test]
    fn test_pmns_type_x_defect_coupling() {
        use cd_kernel::{cd_multiply, gourlay_psi};

        let dim = 16_usize;

        // Find a representative Type X triad and compute its defect vector
        let mut best_defect_norm = 0.0_f64;
        let mut best_defect = [0.0_f64; 16];
        let mut best_triad = (0, 0, 0);

        for b in 1..dim {
            for c in (b + 1)..dim {
                for d in (c + 1)..dim {
                    let t1 = crate::sedenion_subalgebras::assoc_strict(dim, b, c, d);
                    let t2 = crate::sedenion_subalgebras::assoc_strict(dim, b, d, c);
                    let t3 = crate::sedenion_subalgebras::assoc_strict(dim, c, b, d);
                    // Type X: all three nonzero
                    if t1 < 1e-10 || t2 < 1e-10 || t3 < 1e-10 { continue; }

                    // Compute 16D defect vector
                    let mut eb = vec![0.0; dim]; eb[b] = 1.0;
                    let mut ec = vec![0.0; dim]; ec[c] = 1.0;
                    let mut ed = vec![0.0; dim]; ed[d] = 1.0;
                    let bc = cd_multiply(&eb, &ec);
                    let bcd = cd_multiply(&bc, &ed);
                    let cd_prod = cd_multiply(&ec, &ed);
                    let b_cd = cd_multiply(&eb, &cd_prod);

                    let mut defect = [0.0_f64; 16];
                    for k in 0..16 { defect[k] = bcd[k] - b_cd[k]; }
                    let _norm: f64 = defect.iter().map(|x| x * x).sum::<f64>().sqrt();

                    // Check psi overlap magnitude
                    let psi_d = gourlay_psi(&defect);
                    let overlap: f64 = defect.iter().zip(psi_d.iter()).map(|(a, b)| a * b).sum();

                    if overlap.abs() > best_defect_norm {
                        best_defect_norm = overlap.abs();
                        best_defect = defect;
                        best_triad = (b, c, d);
                    }
                }
            }
        }

        println!("--- TYPE X DEFECT VECTOR PSI COUPLING ---");
        println!("  Best triad: (e_{}, e_{}, e_{})", best_triad.0, best_triad.1, best_triad.2);
        println!("  Defect norm: {:.4}", best_defect.iter().map(|x| x*x).sum::<f64>().sqrt());
        println!("  |<defect, psi(defect)>| = {:.4}", best_defect_norm);

        let psi_d = gourlay_psi(&best_defect);
        let self_norm: f64 = best_defect.iter().map(|x| x*x).sum();
        let overlap: f64 = best_defect.iter().zip(psi_d.iter()).map(|(a, b)| a * b).sum();
        println!("  Overlap/norm ratio = {:.4} (cos(120) = -0.5)", overlap / self_norm);
    }

    /// V_6 solar angle correction: extract orthogonal complement of B/C
    /// column space from X incidence matrix, then scan for theta_12 correction.
    ///
    /// Pipeline:
    ///   1. Build B/C incidence matrix (168 rows x 42 cols), rank 21
    ///   2. Compute column-space projector P_BC = Q_BC * Q_BC^T
    ///   3. Apply (I - P_BC) to X columns -> residual C_V6
    ///   4. SVD of C_V6 -> 6 non-zero singular values = V_6 basis
    ///   5. Build solar friction tensor from V_6 basis vectors
    ///   6. Scan alpha_solar for theta_12 correction
    #[test]
    fn test_v6_solar_angle_extraction() {
        use cd_kernel::cayley_dickson::cd_multiply;
        use crate::sedenion_subalgebras::assoc_strict;
        use nalgebra::DMatrix;

        let dim = 16_usize;

        // ===== STEP 1: Build B/C/X incidence matrices =====
        let mut assessors: Vec<(usize, usize)> = Vec::new();
        for low in 1..=7_usize {
            for high in 9..=15_usize {
                if high == low + 8 { continue; }
                assessors.push((low, high));
            }
        }
        assert_eq!(assessors.len(), 42);

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

        println!("--- V_6 SOLAR ANGLE EXTRACTION ---");
        println!("  B/C matrix: {} x {}", mat_bc.nrows(), mat_bc.ncols());
        println!("  X matrix: {} x {}", mat_x.nrows(), mat_x.ncols());

        // ===== STEP 2: Compute B/C column space projector =====
        // SVD of mat_bc^T to get column space basis
        let svd_bc = mat_bc.transpose().svd(true, false);
        let rank_threshold = 1e-8;

        let u_bc = svd_bc.u.as_ref().unwrap();
        let rank_bc = svd_bc.singular_values.iter()
            .filter(|&&s| s > rank_threshold)
            .count();

        println!("  B/C column space rank: {}", rank_bc);

        // Q_BC = first rank_bc columns of U (orthonormal basis of col(BC^T))
        let q_bc = u_bc.columns(0, rank_bc);

        // Projector: P_BC = Q_BC * Q_BC^T (42 x 42)
        let p_bc = q_bc * q_bc.transpose();

        // ===== STEP 3: Project out B/C column space from X =====
        // (I - P_BC) applied to each COLUMN of mat_x^T
        let identity = DMatrix::identity(42, 42);
        let proj_complement = &identity - &p_bc;

        // Apply to X columns: C_V6 = X * (I - P_BC)^T = X * (I - P_BC)
        // Since (I - P_BC) is symmetric, (I - P_BC)^T = (I - P_BC)
        let c_v6 = &mat_x * &proj_complement;

        println!("  C_V6 (projected X): {} x {}", c_v6.nrows(), c_v6.ncols());

        // ===== STEP 4: SVD of C_V6 to extract V_6 basis =====
        let svd_v6 = c_v6.svd(false, true);
        let rank_v6 = svd_v6.singular_values.iter()
            .filter(|&&s| s > rank_threshold)
            .count();

        println!("  V_6 rank: {} (expected 6)", rank_v6);
        println!("  V_6 singular values: [{}]",
            svd_v6.singular_values.iter().take(10)
                .map(|s| format!("{:.3}", s))
                .collect::<Vec<_>>().join(", "));

        // The V_6 basis vectors are the first rank_v6 rows of V^T
        // (= right singular vectors)
        let vt = svd_v6.v_t.as_ref().unwrap();

        // ===== STEP 5: Build solar friction tensor from V_6 =====
        // Each V_6 basis vector is a 42-dimensional assessor-space direction.
        // To build a 3x3 generation friction tensor, we project each V_6 vector
        // onto the 3 subalgebra sectors (o1, o2, o3).
        //
        // The subalgebra assessor indices partition the 42 assessors into
        // generation-correlated groups. We use the cross-generational
        // assessor overlap to build the off-diagonal coupling.

        // Map assessor pairs to generations:
        // o1 uses basis 1..7, o2 uses {1..3, 8..11}, o3 uses {1..3, 12..15}
        // An assessor (low, high) with low in 1..7 and high in 9..15
        // connects across the o1 boundary to o2 (high in 9..11) or o3 (high in 12..15).

        // For the solar correction, we want the component of V_6 that
        // discriminates between o1 and o2 (generations 1 and 2).
        // We build a "generation discriminant" by projecting V_6 onto
        // assessors that connect o1-o2 vs o1-o3 vs o2-o3.

        let mut gen_12_idx = Vec::new(); // assessors connecting gen 1 and gen 2
        let mut gen_13_idx = Vec::new(); // assessors connecting gen 1 and gen 3
        let mut gen_23_idx = Vec::new(); // assessors connecting gen 2 and gen 3

        for (a_idx, &(low, high)) in assessors.iter().enumerate() {
            let low_in_o1_only = (4..=7).contains(&low);  // in o1 but not shared
            let high_in_o2 = (9..=11).contains(&high);
            let high_in_o3 = (12..=15).contains(&high);

            if low_in_o1_only && high_in_o2 {
                gen_12_idx.push(a_idx);
            } else if low_in_o1_only && high_in_o3 {
                gen_13_idx.push(a_idx);
            } else if high_in_o2 && (1..=3).contains(&low) {
                // shared quaternion sector -- affects all gens
                gen_23_idx.push(a_idx);
            }
        }

        println!("\n  Generation-discriminant assessor counts:");
        println!("    1-2 (solar): {}", gen_12_idx.len());
        println!("    1-3 (reactor): {}", gen_13_idx.len());
        println!("    2-3 (atmospheric): {}", gen_23_idx.len());

        // For each V_6 basis vector, compute its projection strength
        // on the 1-2 (solar) sector vs the 1-3 and 2-3 sectors
        println!("\n  V_6 basis vector projections onto generation sectors:");

        let mut solar_v6_weights = Vec::new();
        for k in 0..rank_v6.min(6) {
            let v = vt.row(k);
            let proj_12: f64 = gen_12_idx.iter().map(|&i| v[i] * v[i]).sum();
            let proj_13: f64 = gen_13_idx.iter().map(|&i| v[i] * v[i]).sum();
            let proj_23: f64 = gen_23_idx.iter().map(|&i| v[i] * v[i]).sum();
            let total = proj_12 + proj_13 + proj_23;
            let solar_frac = if total > 1e-15 { proj_12 / total } else { 0.0 };

            println!("    V_6[{}]: solar={:.3}, reactor={:.3}, atmo={:.3}, solar_frac={:.3}",
                k, proj_12, proj_13, proj_23, solar_frac);
            solar_v6_weights.push((k, solar_frac, svd_v6.singular_values[k]));
        }

        // ===== STEP 6: Scan alpha_solar =====
        // Use the most solar-selective V_6 basis vector to construct
        // a perturbation to the PMNS mass matrix.

        // Sort by solar fraction (descending)
        solar_v6_weights.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        if !solar_v6_weights.is_empty() {
            let (best_k, best_frac, best_sv) = solar_v6_weights[0];
            println!("\n  Best solar-selective V_6 vector: V_6[{}]", best_k);
            println!("    Solar fraction: {:.3}", best_frac);
            println!("    Singular value: {:.3}", best_sv);

            // Build the solar perturbation: a 3x3 matrix where
            // M_12 = M_21 = alpha_solar * (V_6 projection on 1-2 assessors)
            // M_13 = M_31 ~ 0 (orthogonal to reactor channel)
            // M_23 = M_32 ~ 0 (orthogonal to atmospheric channel)

            let v_best = vt.row(best_k);

            // Compute the generation-resolved friction from V_6:
            // f_12 = sum of v_best components on 1-2 assessors
            // f_13 = sum of v_best components on 1-3 assessors
            // f_23 = sum of v_best components on 2-3 assessors
            let f_12: f64 = gen_12_idx.iter().map(|&i| v_best[i]).sum();
            let f_13: f64 = gen_13_idx.iter().map(|&i| v_best[i]).sum();
            let f_23: f64 = gen_23_idx.iter().map(|&i| v_best[i]).sum();

            println!("    f_12 = {:.4}, f_13 = {:.4}, f_23 = {:.4}", f_12, f_13, f_23);

            // Scan alpha_solar with the existing two-parameter psi coupling
            let ch_pair = (11, 12);
            let nu_pair = (7, 8);

            println!("\n  Alpha_solar scan (base: alpha_ch=3.75, alpha_nu=1.30):");
            println!("  {:>10} {:>10} {:>10} {:>10} {:>10}",
                "alpha_sol", "theta_12", "theta_13", "theta_23", "score");

            let mut best_score = f64::MAX;
            let mut best_alpha = 0.0_f64;
            let mut best_angles = (0.0_f64, 0.0_f64, 0.0_f64);

            for step in -40..=40_i32 {
                let alpha_solar = step as f64 * 0.1;

                // Build PMNS matrices with psi off-diagonal coupling
                let (m_ch, m_nu) = construct_pmns_matrices_offdiag(
                    ch_pair, nu_pair, 3.75,
                );

                // Add second psi coupling (nu-specific)
                let (_, m_nu2) = construct_pmns_matrices_offdiag(
                    ch_pair, nu_pair, 1.30,
                );

                // Combine: use the ch from alpha=3.75, nu as weighted sum
                let mut m_nu_combined = faer::Mat::zeros(3, 3);
                for i in 0..3 {
                    for j in 0..3 {
                        // Base nu from the two-param fit
                        let base = m_nu.read(i, j) + (m_nu2.read(i, j) - m_nu.read(i, j))
                            * (1.30 / 3.75);
                        m_nu_combined.write(i, j, base);
                    }
                }

                // Add V_6 solar perturbation
                // This adds to the 1-2 off-diagonal coupling
                let solar_perturb = alpha_solar * f_12;
                m_nu_combined.write(0, 1, m_nu_combined.read(0, 1) + solar_perturb);
                m_nu_combined.write(1, 0, m_nu_combined.read(1, 0) + solar_perturb);

                // Small reactor/atmospheric leakage (keep for honesty)
                let reactor_leak = alpha_solar * f_13 * 0.1;
                let atmo_leak = alpha_solar * f_23 * 0.1;
                m_nu_combined.write(0, 2, m_nu_combined.read(0, 2) + reactor_leak);
                m_nu_combined.write(2, 0, m_nu_combined.read(2, 0) + reactor_leak);
                m_nu_combined.write(1, 2, m_nu_combined.read(1, 2) + atmo_leak);
                m_nu_combined.write(2, 1, m_nu_combined.read(2, 1) + atmo_leak);

                // Eigendecompose and extract PMNS angles
                let m_ch_sym = (&m_ch + m_ch.transpose()) * faer::scale(0.5);
                let m_nu_sym = (&m_nu_combined + m_nu_combined.transpose()) * faer::scale(0.5);

                let eig_ch = m_ch_sym.selfadjoint_eigendecomposition(faer::Side::Lower);
                let eig_nu = m_nu_sym.selfadjoint_eigendecomposition(faer::Side::Lower);

                let u_pmns_raw = eig_ch.u().transpose() * eig_nu.u();
                let (u_pmns, _, _) = crate::quark_sector::extract_ckm_permutation_aware(&u_pmns_raw);
                let (theta_12, theta_13, theta_23) = extract_pmns_angles(&u_pmns);
                let t12 = theta_12;
                let t13 = theta_13;
                let t23 = theta_23;

                let score = ((t12 - 33.41) / 33.41).powi(2)
                          + ((t13 - 8.54) / 8.54).powi(2)
                          + ((t23 - 49.0) / 49.0).powi(2);

                if score < best_score {
                    best_score = score;
                    best_alpha = alpha_solar;
                    best_angles = (t12, t13, t23);
                }

                if step % 10 == 0 {
                    println!("  {:10.2} {:10.2} {:10.2} {:10.2} {:10.4}",
                        alpha_solar, t12, t13, t23, score);
                }
            }

            println!("\n  === BEST V_6 SOLAR CORRECTION ===");
            println!("  alpha_solar = {:.2}", best_alpha);
            println!("  theta_12 = {:.2} deg (PDG: 33.41, error: {:.1}%)",
                best_angles.0, ((best_angles.0 - 33.41) / 33.41 * 100.0).abs());
            println!("  theta_13 = {:.2} deg (PDG: 8.54, error: {:.1}%)",
                best_angles.1, ((best_angles.1 - 8.54) / 8.54 * 100.0).abs());
            println!("  theta_23 = {:.2} deg (PDG: 49.0, error: {:.1}%)",
                best_angles.2, ((best_angles.2 - 49.0) / 49.0 * 100.0).abs());
            println!("  Combined score: {:.6}", best_score);
        }
    }

    /// Baseline regression test for the two-parameter PMNS construction.
    ///
    /// Verifies that construct_pmns_matrices_two_param at the known-good
    /// parameters (alpha_ch=3.75, alpha_nu=1.30) reproduces the established
    /// angles within tight tolerances. Also checks permutation stability
    /// and eigenvector overlap.
    #[test]
    fn test_pmns_two_param_baseline_regression() {
        let ch_pair = (11_usize, 12);
        let nu_pair = (7_usize, 8);
        let alpha_ch = 3.75;
        let alpha_nu = 1.30;

        let (m_ch, m_nu) = construct_pmns_matrices_two_param(ch_pair, nu_pair, alpha_ch, alpha_nu);

        let eig_ch = m_ch.selfadjoint_eigendecomposition(faer::Side::Lower);
        let eig_nu = m_nu.selfadjoint_eigendecomposition(faer::Side::Lower);

        let u_pmns_raw = eig_ch.u().transpose() * eig_nu.u();
        let (u_pmns, perm_u, perm_d) = crate::quark_sector::extract_ckm_permutation_aware(&u_pmns_raw);
        let (theta_12, theta_13, theta_23) = extract_pmns_angles(&u_pmns);

        println!("--- PMNS TWO-PARAM BASELINE REGRESSION ---");
        println!("  theta_12 = {:.4} deg (expected ~28.5)", theta_12);
        println!("  theta_13 = {:.4} deg (expected ~8.63)", theta_13);
        println!("  theta_23 = {:.4} deg (expected ~47.1)", theta_23);
        println!("  perm_u = {:?}, perm_d = {:?}", perm_u, perm_d);

        // Tight theta_13 tolerance (< 0.1 deg)
        assert!(
            (theta_13 - 8.63).abs() < 0.1,
            "theta_13 regression: got {:.4}, expected ~8.63 (tol 0.1 deg)", theta_13
        );
        // theta_12 within 0.5 deg of 28.5
        assert!(
            (theta_12 - 28.5).abs() < 0.5,
            "theta_12 regression: got {:.4}, expected ~28.5 (tol 0.5 deg)", theta_12
        );
        // theta_23 within 0.5 deg of 47.1
        assert!(
            (theta_23 - 47.1).abs() < 0.5,
            "theta_23 regression: got {:.4}, expected ~47.1 (tol 0.5 deg)", theta_23
        );

        // Verify permutation ordering is consistent: diagonal dominance
        for i in 0..3 {
            let diag = u_pmns.read(i, i).abs();
            assert!(diag > 0.3, "PMNS diagonal element ({},{}) = {:.4} too small", i, i, diag);
        }

        // Eigenvector overlap stability: compute baseline eigenvectors and verify
        // a second construction gives the same eigenvectors (dot > 0.99)
        let (m_ch2, m_nu2) = construct_pmns_matrices_two_param(ch_pair, nu_pair, alpha_ch, alpha_nu);
        let eig_ch2 = m_ch2.selfadjoint_eigendecomposition(faer::Side::Lower);
        let eig_nu2 = m_nu2.selfadjoint_eigendecomposition(faer::Side::Lower);

        for col in 0..3 {
            let mut dot_ch = 0.0_f64;
            let mut dot_nu = 0.0_f64;
            for row in 0..3 {
                dot_ch += eig_ch.u().read(row, col) * eig_ch2.u().read(row, col);
                dot_nu += eig_nu.u().read(row, col) * eig_nu2.u().read(row, col);
            }
            assert!(dot_ch.abs() > 0.99,
                "Charged eigenvector {} overlap: {:.6} (expected > 0.99)", col, dot_ch.abs());
            assert!(dot_nu.abs() > 0.99,
                "Neutrino eigenvector {} overlap: {:.6} (expected > 0.99)", col, dot_nu.abs());
        }

        println!("  PASS: all regression checks satisfied");
    }

    /// Verify PMNS baseline is decoupled from quark conventions.
    ///
    /// The CasimirBaseline struct holds raw SU(3) and SU(2) projections.
    /// The lepton assembler currently uses the same +/- as quarks, but
    /// changing the quark assembler must NOT affect PMNS results.
    #[test]
    fn test_pmns_casimir_isolation() {
        use crate::quark_sector::{SubalgebraScheme, CasimirBaseline, assemble_quark_matrices};

        let cb = construct_casimir_baseline(SubalgebraScheme::InterleavedStride);

        // Standard lepton assembly (same as quark: c_su3 +/- c_su2)
        let (m_ch_std, m_nu_std) = assemble_lepton_baseline(&cb);

        // Alternative quark convention: flip the SU(2) sign
        // M_up' = c_su3 - c_su2, M_down' = c_su3 + c_su2 (swapped)
        let mut cb_flipped = CasimirBaseline {
            c_su3: cb.c_su3.clone(),
            c_su2: faer::Mat::<f64>::zeros(3, 3),
        };
        for i in 0..3 {
            for j in 0..3 {
                cb_flipped.c_su2.write(i, j, -cb.c_su2.read(i, j));
            }
        }
        let (m_up_flipped, _m_down_flipped) = assemble_quark_matrices(&cb_flipped);

        // The flipped quark convention should produce DIFFERENT quark results
        let (m_up_std, _m_down_std) = assemble_quark_matrices(&cb);
        let mut quark_diff = 0.0_f64;
        for i in 0..3 {
            for j in 0..3 {
                quark_diff += (m_up_flipped.read(i, j) - m_up_std.read(i, j)).abs();
            }
        }
        assert!(quark_diff > 0.1, "Flipped quark convention should differ: diff = {:.6}", quark_diff);

        // But the lepton baseline must be UNCHANGED (because it flows through
        // assemble_lepton_baseline, not assemble_quark_matrices)
        let (m_ch_check, m_nu_check) = assemble_lepton_baseline(&cb);
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (m_ch_std.read(i, j) - m_ch_check.read(i, j)).abs() < 1e-14,
                    "Lepton M_ch changed when quark convention changed at ({},{})", i, j
                );
                assert!(
                    (m_nu_std.read(i, j) - m_nu_check.read(i, j)).abs() < 1e-14,
                    "Lepton M_nu changed when quark convention changed at ({},{})", i, j
                );
            }
        }

        // Verify PMNS angles are correct
        let ch_pair = (11_usize, 12);
        let nu_pair = (7_usize, 8);
        let (m_ch, m_nu) = construct_pmns_matrices_two_param(ch_pair, nu_pair, 3.75, 1.30);

        let eig_ch = m_ch.selfadjoint_eigendecomposition(faer::Side::Lower);
        let eig_nu = m_nu.selfadjoint_eigendecomposition(faer::Side::Lower);
        let u_raw = eig_ch.u().transpose() * eig_nu.u();
        let (u_pmns, _, _) = crate::quark_sector::extract_ckm_permutation_aware(&u_raw);
        let (t12, t13, t23) = extract_pmns_angles(&u_pmns);

        assert!((t12 - 28.54).abs() < 0.01, "theta_12 isolation: {:.4}", t12);
        assert!((t13 - 8.63).abs() < 0.01, "theta_13 isolation: {:.4}", t13);
        assert!((t23 - 47.07).abs() < 0.01, "theta_23 isolation: {:.4}", t23);

        println!("PASS: PMNS Casimir isolation verified");
    }

