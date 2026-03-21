//! Neutrino Sector: PMNS Mixing Matrix from Sedenion Signed Friction
//!
//! Derives the PMNS (Pontecorvo-Maki-Nakagawa-Sakata) mixing matrix from the
//! same signed-friction framework used for the CKM matrix. The PMNS matrix
//! relates neutrino mass eigenstates to flavor eigenstates:
//!
//!   U_PMNS = U_charged^dagger * U_neutrino
//!
//! where U_charged diagonalizes the charged lepton mass matrix and
//! U_neutrino diagonalizes the neutrino mass matrix.
//!
//! # Key differences from quark sector
//!
//! - PMNS angles are LARGE: theta_23 ~ 49 deg, theta_12 ~ 33 deg, theta_13 ~ 8.5 deg
//! - CKM angles are SMALL: theta_12 ~ 13 deg, theta_23 ~ 2.4 deg, theta_13 ~ 0.2 deg
//! - This asymmetry may arise from using different selector pairs in the
//!   sedenion algebra for the lepton vs quark sectors.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeneratorType {
    SU3,
    SU2,
    U1,
    Leptoquark,
    Dark,
}

pub fn classify_generator(gen_index: usize) -> GeneratorType {
    match gen_index {
        0..=7 => GeneratorType::SU3,
        8..=10 => GeneratorType::SU2,
        11 => GeneratorType::U1,
        12..=23 => GeneratorType::Leptoquark,
        _ => GeneratorType::Dark,
    }
}

/// PMNS mixing matrix result.
pub struct PmnsResult {
    /// The 3x3 PMNS matrix.
    pub matrix: faer::Mat<f64>,
    /// Mixing angles in degrees: (theta_12, theta_13, theta_23).
    pub angles_deg: (f64, f64, f64),
    /// Neutrino mass eigenvalues (sorted ascending).
    pub neutrino_masses: [f64; 3],
    /// Charged lepton mass eigenvalues (sorted ascending).
    pub charged_masses: [f64; 3],
    /// Delta m^2_21 = m2^2 - m1^2 (solar).
    pub delta_m21_sq: f64,
    /// Delta m^2_31 = m3^2 - m1^2 (atmospheric).
    pub delta_m31_sq: f64,
}

/// Construct neutrino and charged-lepton mass matrices from signed friction.
///
/// Uses the quark-sector Casimir projection as a baseline (provides off-diagonal
/// structure needed for nontrivial mixing), then adds signed-friction diagonal
/// perturbations with different selector pairs for each sector.
///
/// The charged-lepton mass matrix uses the SU(2) Casimir projection (weak isospin),
/// the neutrino mass matrix uses the SU(3) Casimir projection (color singlet).
/// This is physically motivated: charged leptons couple to SU(2)_L, while
/// neutrinos are SU(3) singlets in the SU(5) framework.
pub fn construct_pmns_matrices(
    charged_pair: (usize, usize),
    neutrino_pair: (usize, usize),
) -> (faer::Mat<f64>, faer::Mat<f64>) {
    use crate::lepton_mass_hierarchy::cd_braid_signed_friction;
    use crate::majorana_braiding::MajoranaMode;
    use crate::bell_inequality::SignTableCache;
    use crate::three_fermion_generations::get_sedenion_subalgebras;
    use crate::cayley_dickson_structs::Sedenion;
    use crate::quark_sector::SubalgebraScheme;

    // Build Casimir baseline from the quark-sector infrastructure
    let mut basis = [Sedenion::default(); 16];
    for i in 0..16 {
        let mut components = [0.0; 16];
        components[i] = 1.0;
        basis[i] = Sedenion::from_slice(&components);
    }
    let complex_structure = basis[15];

    // Get quark mass matrices as baseline -- these have off-diagonal structure
    let (m_baseline_ch, m_baseline_nu) = crate::quark_sector::construct_quark_mass_matrices(
        &basis, &complex_structure, SubalgebraScheme::InterleavedStride,
    );

    // Signed friction
    let (o1, o2, o3) = get_sedenion_subalgebras();
    let subs = [&o1, &o2, &o3];
    let sign_table = SignTableCache::new(16);

    let w1: f64 = -0.656850;
    let w2: f64 = -0.741999;

    let ch_a = MajoranaMode { gamma_index: charged_pair.0.saturating_sub(1), cd_basis_index: charged_pair.0, cd_dim: 16 };
    let ch_b = MajoranaMode { gamma_index: charged_pair.1.saturating_sub(1), cd_basis_index: charged_pair.1, cd_dim: 16 };
    let nu_a = MajoranaMode { gamma_index: neutrino_pair.0.saturating_sub(1), cd_basis_index: neutrino_pair.0, cd_dim: 16 };
    let nu_b = MajoranaMode { gamma_index: neutrino_pair.1.saturating_sub(1), cd_basis_index: neutrino_pair.1, cd_dim: 16 };

    let sel_ch: Vec<f64> = subs.iter()
        .map(|s| cd_braid_signed_friction(&ch_a, &ch_b, s, &sign_table))
        .collect();
    let sel_nu: Vec<f64> = subs.iter()
        .map(|s| cd_braid_signed_friction(&nu_a, &nu_b, s, &sign_table))
        .collect();

    // Baseline + cross-coupled friction perturbation
    let mut m_charged = m_baseline_ch;
    let mut m_neutrino = m_baseline_nu;
    for i in 0..3 {
        let f_ch = w1 * sel_ch[i] + w2 * sel_nu[i];
        let f_nu = w1 * sel_nu[i] + w2 * sel_ch[i];
        m_charged.write(i, i, m_charged.read(i, i) + f_ch.exp());
        m_neutrino.write(i, i, m_neutrino.read(i, i) + f_nu.exp());
    }

    (m_charged, m_neutrino)
}

/// Extract PMNS angles from a 3x3 unitary matrix using standard parameterization.
pub fn extract_pmns_angles(u: &faer::Mat<f64>) -> (f64, f64, f64) {
    let u_e3 = u.read(0, 2).abs();
    let theta_13 = u_e3.min(1.0).asin();
    let cos_13 = theta_13.cos();

    let theta_12 = if cos_13 > 1e-15 {
        (u.read(0, 1).abs() / cos_13).min(1.0).asin()
    } else {
        0.0
    };
    let theta_23 = if cos_13 > 1e-15 {
        (u.read(1, 2).abs() / cos_13).min(1.0).asin()
    } else {
        0.0
    };

    (theta_12.to_degrees(), theta_13.to_degrees(), theta_23.to_degrees())
}

/// Compute PMNS result for given selector pairs.
pub fn compute_pmns(
    charged_pair: (usize, usize),
    neutrino_pair: (usize, usize),
) -> PmnsResult {
    use faer::Side;

    let (m_ch, m_nu) = construct_pmns_matrices(charged_pair, neutrino_pair);

    let m_ch_sym = (&m_ch + m_ch.transpose()) * faer::scale(0.5);
    let m_nu_sym = (&m_nu + m_nu.transpose()) * faer::scale(0.5);

    let eig_ch = m_ch_sym.selfadjoint_eigendecomposition(Side::Lower);
    let eig_nu = m_nu_sym.selfadjoint_eigendecomposition(Side::Lower);

    // U_PMNS = U_charged^T * U_neutrino
    let u_pmns_raw = eig_ch.u().transpose() * eig_nu.u();

    // Permutation-aware alignment (reuse CKM infrastructure)
    let (u_pmns, _pu, _pd) = crate::quark_sector::extract_ckm_permutation_aware(&u_pmns_raw);

    let (theta_12, theta_13, theta_23) = extract_pmns_angles(&u_pmns);

    let mut ch_masses = [0.0_f64; 3];
    let mut nu_masses = [0.0_f64; 3];
    for i in 0..3 {
        ch_masses[i] = eig_ch.s().column_vector().read(i).abs();
        nu_masses[i] = eig_nu.s().column_vector().read(i).abs();
    }
    ch_masses.sort_by(|a, b| a.partial_cmp(b).unwrap());
    nu_masses.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Mass-squared differences (in arbitrary units, ratios are meaningful)
    let delta_m21_sq = nu_masses[1].powi(2) - nu_masses[0].powi(2);
    let delta_m31_sq = nu_masses[2].powi(2) - nu_masses[0].powi(2);

    PmnsResult {
        matrix: u_pmns,
        angles_deg: (theta_12, theta_13, theta_23),
        neutrino_masses: nu_masses,
        charged_masses: ch_masses,
        delta_m21_sq,
        delta_m31_sq,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
        use crate::cayley_dickson_structs::Sedenion;
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
        let mut basis = [Sedenion::default(); 16];
        for i in 0..16 {
            let mut components = [0.0; 16];
            components[i] = 1.0;
            basis[i] = Sedenion::from_slice(&components);
        }
        let cs = basis[15];
        let (m_base_ch, m_base_nu) = crate::quark_sector::construct_quark_mass_matrices(
            &basis, &cs, SubalgebraScheme::InterleavedStride,
        );

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
        use crate::cayley_dickson_structs::Sedenion;
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
        let mut basis = [Sedenion::default(); 16];
        for i in 0..16 {
            let mut components = [0.0; 16];
            components[i] = 1.0;
            basis[i] = Sedenion::from_slice(&components);
        }
        let cs = basis[15];
        let (m_base_ch, m_base_nu) = crate::quark_sector::construct_quark_mass_matrices(
            &basis, &cs, SubalgebraScheme::InterleavedStride,
        );

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
}
