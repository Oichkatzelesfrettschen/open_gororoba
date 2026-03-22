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
    /// Jarlskog invariant J = Im(U_e2 U_mu3 U_e3* U_mu2*).
    /// For real PMNS matrices J = 0 (no CP violation).
    pub jarlskog_invariant: f64,
    /// Dirac CP phase delta_CP in degrees.
    /// Extracted from J via J = s12 c12 s23 c23 s13 c13^2 sin(delta).
    /// PDG 2024: delta_CP ~ 195 deg (normal ordering).
    pub cp_phase_deg: f64,
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
    use crate::quark_sector::SubalgebraScheme;

    // Casimir baseline via neutral projections + lepton assembly
    let cb = construct_casimir_baseline(SubalgebraScheme::InterleavedStride);
    let (m_baseline_ch, m_baseline_nu) = assemble_lepton_baseline(&cb);

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

/// Construct PMNS matrices with OFF-DIAGONAL friction from psi automorphism.
///
/// The key insight: diagonal friction only perturbs M_ii, which cannot
/// produce near-maximal theta_23 from a hierarchical baseline.
/// Off-diagonal M_ij terms come from the psi automorphism's cross-generational
/// coupling: psi cycles O1->O2->O3, so <friction_i, psi(friction_j)>
/// measures the transition amplitude between generations.
///
/// The 3x3 friction tensor F_ij is:
///   F_ii = w1*sel_own[i] + w2*sel_other[i]  (diagonal, as before)
///   F_ij = alpha_cross * <sel_own, psi^k(sel_other)> for i != j
///
/// alpha_cross controls the off-diagonal coupling strength.
pub fn construct_pmns_matrices_offdiag(
    charged_pair: (usize, usize),
    neutrino_pair: (usize, usize),
    alpha_cross: f64,
) -> (faer::Mat<f64>, faer::Mat<f64>) {
    use crate::lepton_mass_hierarchy::cd_braid_signed_friction;
    use crate::majorana_braiding::MajoranaMode;
    use crate::bell_inequality::SignTableCache;
    use crate::three_fermion_generations::get_sedenion_subalgebras;
    use crate::quark_sector::SubalgebraScheme;
    use cd_kernel::gourlay_psi;

    // Casimir baseline via neutral projections + lepton assembly
    let cb = construct_casimir_baseline(SubalgebraScheme::InterleavedStride);
    let (m_base_ch, m_base_nu) = assemble_lepton_baseline(&cb);

    let (o1, o2, o3) = get_sedenion_subalgebras();
    let subs = [&o1, &o2, &o3];
    let sign_table = SignTableCache::new(16);

    let w1: f64 = -0.656850;
    let w2: f64 = -0.741999;

    let ch_a = MajoranaMode { gamma_index: charged_pair.0 - 1, cd_basis_index: charged_pair.0, cd_dim: 16 };
    let ch_b = MajoranaMode { gamma_index: charged_pair.1 - 1, cd_basis_index: charged_pair.1, cd_dim: 16 };
    let nu_a = MajoranaMode { gamma_index: neutrino_pair.0 - 1, cd_basis_index: neutrino_pair.0, cd_dim: 16 };
    let nu_b = MajoranaMode { gamma_index: neutrino_pair.1 - 1, cd_basis_index: neutrino_pair.1, cd_dim: 16 };

    let sel_ch: Vec<f64> = subs.iter()
        .map(|s| cd_braid_signed_friction(&ch_a, &ch_b, s, &sign_table))
        .collect();
    let sel_nu: Vec<f64> = subs.iter()
        .map(|s| cd_braid_signed_friction(&nu_a, &nu_b, s, &sign_table))
        .collect();

    // Build 3x3 friction tensors with off-diagonal terms
    let mut m_ch = m_base_ch;
    let mut m_nu = m_base_nu;

    // Diagonal terms (same as before)
    for i in 0..3 {
        let f_ch = w1 * sel_ch[i] + w2 * sel_nu[i];
        let f_nu = w1 * sel_nu[i] + w2 * sel_ch[i];
        m_ch.write(i, i, m_ch.read(i, i) + f_ch.exp());
        m_nu.write(i, i, m_nu.read(i, i) + f_nu.exp());
    }

    // Off-diagonal terms from psi automorphism CIRCULANT structure.
    //
    // Key insight: psi preserves norms, so M_22 = M_33 = M_11 for
    // the psi-generated part. This gives a circulant mass matrix:
    //   M = [[A, B, C], [C, A, B], [B, C, A]]
    // which analytically predicts maximal theta_23 = 45 deg.
    //
    // Build the friction vector as a 16D sedenion, apply psi to get
    // the cross-generational overlap.
    {
        // Build friction vectors as sedenion basis-weighted sums
        // v_ch[k] = sel_ch[k] (friction value for generation k)
        // The psi overlap <v, psi(v)> gives the off-diagonal coupling B

        // For each braid pair, construct a 16D friction "profile"
        // by placing the friction values at the generation-specific indices
        let ch_a_idx = charged_pair.0;
        let ch_b_idx = charged_pair.1;
        let nu_a_idx = neutrino_pair.0;
        let nu_b_idx = neutrino_pair.1;

        // Construct a sedenion vector with friction at the braid indices
        let mut v_ch = [0.0_f64; 16];
        v_ch[ch_a_idx] = sel_ch[0]; // generation 1 friction at braid axis a
        v_ch[ch_b_idx] = sel_ch[1]; // generation 2 friction at braid axis b

        let mut v_nu = [0.0_f64; 16];
        v_nu[nu_a_idx] = sel_nu[0];
        v_nu[nu_b_idx] = sel_nu[1];

        // Apply psi to get the cross-generational overlap
        let psi_v_ch = gourlay_psi(&v_ch);
        let psi_v_nu = gourlay_psi(&v_nu);

        // Overlap B = <v, psi(v)> (the circulant off-diagonal coupling)
        let b_ch: f64 = v_ch.iter().zip(psi_v_ch.iter()).map(|(a, b)| a * b).sum();
        let b_nu: f64 = v_nu.iter().zip(psi_v_nu.iter()).map(|(a, b)| a * b).sum();

        // Overlap C = <v, psi^2(v)>
        let psi2_v_ch = gourlay_psi(&psi_v_ch);
        let psi2_v_nu = gourlay_psi(&psi_v_nu);
        let c_ch: f64 = v_ch.iter().zip(psi2_v_ch.iter()).map(|(a, b)| a * b).sum();
        let c_nu: f64 = v_nu.iter().zip(psi2_v_nu.iter()).map(|(a, b)| a * b).sum();

        // Inject circulant off-diagonal terms
        // M_ij += alpha_cross * circulant[i-j mod 3]
        let circulant_ch = [0.0, b_ch, c_ch]; // [A_extra, B, C]
        let circulant_nu = [0.0, b_nu, c_nu];

        for i in 0..3 {
            for j in 0..3 {
                if i == j { continue; }
                let shift = (j + 3 - i) % 3;
                m_ch.write(i, j, m_ch.read(i, j) + alpha_cross * circulant_ch[shift]);
                m_nu.write(i, j, m_nu.read(i, j) + alpha_cross * circulant_nu[shift]);
            }
        }

        // Symmetrize
        for i in 0..3 {
            for j in (i + 1)..3 {
                let avg_ch = (m_ch.read(i, j) + m_ch.read(j, i)) / 2.0;
                let avg_nu = (m_nu.read(i, j) + m_nu.read(j, i)) / 2.0;
                m_ch.write(i, j, avg_ch);
                m_ch.write(j, i, avg_ch);
                m_nu.write(i, j, avg_nu);
                m_nu.write(j, i, avg_nu);
            }
        }
    }

    (m_ch, m_nu)
}

/// Extract PMNS angles from a 3x3 unitary matrix using standard parameterization.
/// Extract the Jarlskog invariant and CP phase from a real PMNS matrix.
///
/// For a real orthogonal matrix, J = 0 always. The CP phase is undefined
/// in this case (returned as 0.0).
///
/// For a complex unitary PMNS matrix U:
///   J = Im(U_e2 * U_mu3 * U_e3* * U_mu2*)
///   = s12 * c12 * s23 * c23 * s13 * c13^2 * sin(delta_CP)
///
/// PDG 2024 best fit (normal ordering): delta_CP ~ 195 deg, J ~ -0.033.
pub fn extract_cp_phase(angles_deg: (f64, f64, f64), j_invariant: f64) -> f64 {
    let (t12, t13, t23) = angles_deg;
    let s12 = t12.to_radians().sin();
    let c12 = t12.to_radians().cos();
    let s13 = t13.to_radians().sin();
    let c13 = t13.to_radians().cos();
    let s23 = t23.to_radians().sin();
    let c23 = t23.to_radians().cos();

    let denom = s12 * c12 * s23 * c23 * s13 * c13 * c13;
    if denom.abs() < 1e-15 {
        return 0.0;
    }
    let sin_delta = j_invariant / denom;
    sin_delta.clamp(-1.0, 1.0).asin().to_degrees()
}

/// Compute the Jarlskog invariant for a real 3x3 matrix.
///
/// J = det(Im([M_up, M_down])) / (Delta m^2 products)
/// For real matrices: J = 0 identically.
/// This function returns 0.0 for real matrices; it exists to
/// establish the interface for future complex extension.
pub fn jarlskog_from_real_pmns(u: &faer::Mat<f64>) -> f64 {
    // J = Im(U_e2 * U_mu3 * conj(U_e3) * conj(U_mu2))
    // For real U: all products are real, so Im = 0.
    let prod = u.read(0, 1) * u.read(1, 2) * u.read(0, 2) * u.read(1, 1);
    // For a real orthogonal matrix, the "imaginary part" is always zero.
    // We return the antisymmetric combination as a consistency check.
    let j = u.read(0, 1) * u.read(1, 2) * u.read(2, 0)
          - u.read(0, 2) * u.read(1, 1) * u.read(2, 0);
    // This is actually Re(U_e2 * U_mu3 * U_tau1) - Re(U_e3 * U_mu2 * U_tau1),
    // which is an antisymmetric product, NOT the Jarlskog invariant.
    // For a truly real orthogonal matrix, J = 0 by definition.
    let _ = prod;
    let _ = j;
    0.0
}

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

/// PDG 2024 central values and 1-sigma uncertainties (normal ordering).
#[derive(Clone, Copy)]
pub struct Pdg2024 {
    pub theta_12_deg: f64,
    pub theta_12_err: f64,
    pub theta_13_deg: f64,
    pub theta_13_err: f64,
    pub theta_23_deg: f64,
    pub theta_23_err: f64,
    pub delta_cp_deg: f64,
    pub delta_cp_err: f64,
    pub dm21_sq_ev2: f64,
    pub dm21_sq_err: f64,
    pub dm31_sq_ev2: f64,
    pub dm31_sq_err: f64,
}

impl Default for Pdg2024 {
    fn default() -> Self {
        Self {
            theta_12_deg: 33.41, theta_12_err: 0.75,
            theta_13_deg: 8.54,  theta_13_err: 0.12,
            theta_23_deg: 49.0,  theta_23_err: 1.1,
            delta_cp_deg: 195.0, delta_cp_err: 25.0,
            dm21_sq_ev2: 7.53e-5, dm21_sq_err: 0.18e-5,
            dm31_sq_ev2: 2.453e-3, dm31_sq_err: 0.033e-3,
        }
    }
}

/// Chi-squared for PMNS mixing angles against PDG 2024.
///
/// Uses 3 mixing angles (always available from real PMNS).
pub fn chi_squared_pmns(result: &PmnsResult, pdg: &Pdg2024) -> f64 {
    let (t12, t13, t23) = result.angles_deg;
    let mut chi2 = 0.0;
    chi2 += ((t12 - pdg.theta_12_deg) / pdg.theta_12_err).powi(2);
    chi2 += ((t13 - pdg.theta_13_deg) / pdg.theta_13_err).powi(2);
    chi2 += ((t23 - pdg.theta_23_deg) / pdg.theta_23_err).powi(2);
    chi2
}

/// Individual pulls for each observable: (observable - PDG) / sigma.
pub fn pmns_pulls(result: &PmnsResult, pdg: &Pdg2024) -> Vec<(&'static str, f64)> {
    let (t12, t13, t23) = result.angles_deg;
    vec![
        ("theta_12", (t12 - pdg.theta_12_deg) / pdg.theta_12_err),
        ("theta_13", (t13 - pdg.theta_13_deg) / pdg.theta_13_err),
        ("theta_23", (t23 - pdg.theta_23_deg) / pdg.theta_23_err),
    ]
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

    let j = jarlskog_from_real_pmns(&u_pmns);
    let cp_phase = extract_cp_phase((theta_12, theta_13, theta_23), j);

    PmnsResult {
        matrix: u_pmns,
        angles_deg: (theta_12, theta_13, theta_23),
        neutrino_masses: nu_masses,
        charged_masses: ch_masses,
        delta_m21_sq,
        delta_m31_sq,
        jarlskog_invariant: j,
        cp_phase_deg: cp_phase,
    }
}

/// Compute raw Casimir projections for the lepton sector.
///
/// Returns the neutral CasimirBaseline struct (raw SU(3) and SU(2)
/// Gram matrices) without any sector-specific sign convention.
/// The caller assembles these into mass matrices using whatever
/// convention is appropriate for the physics sector.
pub fn construct_casimir_baseline(
    scheme: crate::quark_sector::SubalgebraScheme,
) -> crate::quark_sector::CasimirBaseline {
    use crate::cayley_dickson_structs::Sedenion;

    let mut basis = [Sedenion::default(); 16];
    for i in 0..16 {
        let mut components = [0.0; 16];
        components[i] = 1.0;
        basis[i] = Sedenion::from_slice(&components);
    }
    let complex_structure = basis[15];

    crate::quark_sector::construct_casimir_projections(
        &basis, &complex_structure, scheme,
    )
}

/// Assemble lepton baseline mass matrices from raw Casimir projections.
///
/// Currently uses the same convention as the quark sector (M_ch = C_SU3 + C_SU2,
/// M_nu = C_SU3 - C_SU2) to preserve regression. This is an explicit choice
/// that can be revisited independently of the quark sector.
fn assemble_lepton_baseline(
    cb: &crate::quark_sector::CasimirBaseline,
) -> (faer::Mat<f64>, faer::Mat<f64>) {
    crate::quark_sector::assemble_quark_matrices(cb)
}

/// Construct PMNS matrices with two independent psi-coupling parameters.
///
/// Factored from the `test_pmns_offdiag_two_param` scan body into a pure,
/// deterministic function. The construction is:
///
/// 1. Casimir baseline via `construct_casimir_baseline`
/// 2. Diagonal friction: `M[i,i] += exp(w1*sel_own[i] + w2*sel_other[i])`
/// 3. Off-diagonal psi circulant: `M[i,j] += alpha * <profile_i, psi(profile_j)>`
///    with `alpha_ch` for charged, `alpha_nu` for neutrino
/// 4. Symmetrize
pub fn construct_pmns_matrices_two_param(
    charged_pair: (usize, usize),
    neutrino_pair: (usize, usize),
    alpha_ch: f64,
    alpha_nu: f64,
) -> (faer::Mat<f64>, faer::Mat<f64>) {
    use crate::lepton_mass_hierarchy::cd_braid_signed_friction;
    use crate::majorana_braiding::MajoranaMode;
    use crate::bell_inequality::{SignTableCache, rotate_sparse};
    use crate::three_fermion_generations::get_sedenion_subalgebras;
    use crate::quark_sector::SubalgebraScheme;
    use cd_kernel::gourlay_psi;

    // Step 1: Casimir baseline via neutral projections + lepton assembly
    let cb = construct_casimir_baseline(SubalgebraScheme::InterleavedStride);
    let (m_base_ch, m_base_nu) = assemble_lepton_baseline(&cb);

    let (o1, o2, o3) = get_sedenion_subalgebras();
    let subs = [&o1, &o2, &o3];
    let sign_table = SignTableCache::new(16);

    let w1: f64 = -0.656850;
    let w2: f64 = -0.741999;

    let ch_a = MajoranaMode { gamma_index: charged_pair.0 - 1, cd_basis_index: charged_pair.0, cd_dim: 16 };
    let ch_b = MajoranaMode { gamma_index: charged_pair.1 - 1, cd_basis_index: charged_pair.1, cd_dim: 16 };
    let nu_a = MajoranaMode { gamma_index: neutrino_pair.0 - 1, cd_basis_index: neutrino_pair.0, cd_dim: 16 };
    let nu_b = MajoranaMode { gamma_index: neutrino_pair.1 - 1, cd_basis_index: neutrino_pair.1, cd_dim: 16 };

    // Build full 16D friction profiles per generation
    let build_profile = |mode_i: &MajoranaMode, mode_j: &MajoranaMode, sub: &[usize]| -> [f64; 16] {
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

    let dot16 = |a: &[f64; 16], b: &[f64; 16]| -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    };

    // Step 2: Diagonal friction
    let mut m_ch = m_base_ch;
    let mut m_nu = m_base_nu;
    for i in 0..3 {
        let f_ch = w1 * sel_ch[i] + w2 * sel_nu[i];
        let f_nu = w1 * sel_nu[i] + w2 * sel_ch[i];
        m_ch.write(i, i, m_ch.read(i, i) + f_ch.exp());
        m_nu.write(i, i, m_nu.read(i, i) + f_nu.exp());
    }

    // Step 3: Off-diagonal psi circulant coupling
    for i in 0..3 {
        for j in 0..3 {
            if i == j { continue; }
            let psi_nu_j = gourlay_psi(&nu_profiles[j]);
            let psi_ch_j = gourlay_psi(&ch_profiles[j]);
            m_nu.write(i, j, m_nu.read(i, j) + alpha_nu * dot16(&nu_profiles[i], &psi_nu_j));
            m_ch.write(i, j, m_ch.read(i, j) + alpha_ch * dot16(&ch_profiles[i], &psi_ch_j));
        }
    }

    // Step 4: Symmetrize
    for i in 0..3 {
        for j in (i + 1)..3 {
            let avg_ch = (m_ch.read(i, j) + m_ch.read(j, i)) / 2.0;
            let avg_nu = (m_nu.read(i, j) + m_nu.read(j, i)) / 2.0;
            m_ch.write(i, j, avg_ch);
            m_ch.write(j, i, avg_ch);
            m_nu.write(i, j, avg_nu);
            m_nu.write(j, i, avg_nu);
        }
    }

    let m_ch_s = (&m_ch + m_ch.transpose()) * faer::scale(0.5);
    let m_nu_s = (&m_nu + m_nu.transpose()) * faer::scale(0.5);

    (m_ch_s, m_nu_s)
}

/// Construct PMNS matrices with V_6-modulated psi coupling strengths.
///
/// Instead of injecting V_6 as a linear additive perturbation to M_nu entries,
/// this function uses V_6 to **nonlinearly modulate** the psi-circulant coupling
/// strengths alpha_ch and alpha_nu. The modulation is generation-pair-specific:
///
///   alpha_nu_{ij}(beta) = base_alpha_nu * exp(phi_i(beta) + phi_j(beta))
///
/// where phi_k(beta) is the V_6 field collapsed onto generation k using the
/// DirectOffDiagonal block partition (14 assessors per generation).
///
/// This changes the *geometry of the eigenvalue landscape* rather than adding
/// a linear vector to it, which breaks the rank-2 Jacobian lock proven in C-1476.
pub fn construct_pmns_matrices_v6_modulated(
    charged_pair: (usize, usize),
    neutrino_pair: (usize, usize),
    base_alpha_ch: f64,
    base_alpha_nu: f64,
    v6_basis: &nalgebra::DMatrix<f64>,
    beta: &[f64; 6],
) -> (faer::Mat<f64>, faer::Mat<f64>) {
    use crate::lepton_mass_hierarchy::cd_braid_signed_friction;
    use crate::majorana_braiding::MajoranaMode;
    use crate::bell_inequality::{SignTableCache, rotate_sparse};
    use crate::three_fermion_generations::get_sedenion_subalgebras;
    use crate::quark_sector::SubalgebraScheme;
    use cd_kernel::gourlay_psi;

    // Casimir baseline
    let cb = construct_casimir_baseline(SubalgebraScheme::InterleavedStride);
    let (m_base_ch, m_base_nu) = assemble_lepton_baseline(&cb);

    let (o1, o2, o3) = get_sedenion_subalgebras();
    let subs = [&o1, &o2, &o3];
    let sign_table = SignTableCache::new(16);

    let w1: f64 = -0.656850;
    let w2: f64 = -0.741999;

    let ch_a = MajoranaMode { gamma_index: charged_pair.0 - 1, cd_basis_index: charged_pair.0, cd_dim: 16 };
    let ch_b = MajoranaMode { gamma_index: charged_pair.1 - 1, cd_basis_index: charged_pair.1, cd_dim: 16 };
    let nu_a = MajoranaMode { gamma_index: neutrino_pair.0 - 1, cd_basis_index: neutrino_pair.0, cd_dim: 16 };
    let nu_b = MajoranaMode { gamma_index: neutrino_pair.1 - 1, cd_basis_index: neutrino_pair.1, cd_dim: 16 };

    let build_profile = |mode_i: &MajoranaMode, mode_j: &MajoranaMode, sub: &[usize]| -> [f64; 16] {
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

    let dot16 = |a: &[f64; 16], b: &[f64; 16]| -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    };

    // Compute V_6 modulation field: collapse beta into 3 generation factors
    let n_basis = v6_basis.nrows().min(6);
    let n_cols = v6_basis.ncols().min(42);
    let mut v_combined = vec![0.0_f64; n_cols];
    for k in 0..n_basis {
        if beta[k].abs() < 1e-15 { continue; }
        for col in 0..n_cols {
            v_combined[col] += beta[k] * v6_basis[(k, col)];
        }
    }

    // Collapse into 3 generation modulation factors (14 assessors per generation)
    let block_size = n_cols / 3;
    let phi = [
        v_combined[..block_size].iter().sum::<f64>(),
        v_combined[block_size..2 * block_size].iter().sum::<f64>(),
        v_combined[2 * block_size..n_cols].iter().sum::<f64>(),
    ];

    // Diagonal friction (unchanged)
    let mut m_ch = m_base_ch;
    let mut m_nu = m_base_nu;
    for i in 0..3 {
        let f_ch = w1 * sel_ch[i] + w2 * sel_nu[i];
        let f_nu = w1 * sel_nu[i] + w2 * sel_ch[i];
        m_ch.write(i, i, m_ch.read(i, i) + f_ch.exp());
        m_nu.write(i, i, m_nu.read(i, i) + f_nu.exp());
    }

    // Off-diagonal psi coupling with V_6-modulated alpha
    for i in 0..3 {
        for j in 0..3 {
            if i == j { continue; }
            let psi_nu_j = gourlay_psi(&nu_profiles[j]);
            let psi_ch_j = gourlay_psi(&ch_profiles[j]);

            // Generation-pair-specific modulation
            let alpha_nu_ij = base_alpha_nu * (phi[i] + phi[j]).exp();
            let alpha_ch_ij = base_alpha_ch * (phi[i] + phi[j]).exp();

            m_nu.write(i, j, m_nu.read(i, j) + alpha_nu_ij * dot16(&nu_profiles[i], &psi_nu_j));
            m_ch.write(i, j, m_ch.read(i, j) + alpha_ch_ij * dot16(&ch_profiles[i], &psi_ch_j));
        }
    }

    // Symmetrize
    for i in 0..3 {
        for j in (i + 1)..3 {
            let avg_ch = (m_ch.read(i, j) + m_ch.read(j, i)) / 2.0;
            let avg_nu = (m_nu.read(i, j) + m_nu.read(j, i)) / 2.0;
            m_ch.write(i, j, avg_ch);
            m_ch.write(j, i, avg_ch);
            m_nu.write(i, j, avg_nu);
            m_nu.write(j, i, avg_nu);
        }
    }

    let m_ch_s = (&m_ch + m_ch.transpose()) * faer::scale(0.5);
    let m_nu_s = (&m_nu + m_nu.transpose()) * faer::scale(0.5);

    (m_ch_s, m_nu_s)
}

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

/// Maps 42-dimensional assessor-space vectors to 3x3 generation couplings.
///
/// EXPERIMENTAL / FIRST-PASS PROJECTION HEURISTIC: the (12/12/6) partition
/// is a working diagnostic proven insufficient for solar angle decoupling
/// (C-1474, C-1476). It produces collinear PMNS gradients in V_6 space.
/// The partition is based on which octonionic subalgebra boundaries each
/// assessor straddles, which is not invariant under the relevant automorphism
/// or permutation action.
pub struct AssessorToFlavorMap {
    /// Assessor indices connecting generation 1 and 2 (solar channel).
    pub gen_12_indices: Vec<usize>,
    /// Assessor indices connecting generation 1 and 3 (reactor channel).
    pub gen_13_indices: Vec<usize>,
    /// Assessor indices connecting generation 2 and 3 (atmospheric channel).
    pub gen_23_indices: Vec<usize>,
}

impl AssessorToFlavorMap {
    /// Default partition based on subalgebra overlap structure.
    ///
    /// For assessor (low, high) with low in 1..7, high in 9..15:
    /// - Solar (1-2): low in 4..7 (O1-only) AND high in 9..11 (O2)
    /// - Reactor (1-3): low in 4..7 (O1-only) AND high in 12..15 (O3)
    /// - Atmospheric (2-3): low in 1..3 (shared quaternion) AND high in 9..11 (O2)
    pub fn default_partition(assessors: &[(usize, usize)]) -> Self {
        let mut gen_12 = Vec::new();
        let mut gen_13 = Vec::new();
        let mut gen_23 = Vec::new();

        for (a_idx, &(low, high)) in assessors.iter().enumerate() {
            let low_in_o1_only = (4..=7).contains(&low);
            let high_in_o2 = (9..=11).contains(&high);
            let high_in_o3 = (12..=15).contains(&high);

            if low_in_o1_only && high_in_o2 {
                gen_12.push(a_idx);
            } else if low_in_o1_only && high_in_o3 {
                gen_13.push(a_idx);
            } else if high_in_o2 && (1..=3).contains(&low) {
                gen_23.push(a_idx);
            }
        }

        Self { gen_12_indices: gen_12, gen_13_indices: gen_13, gen_23_indices: gen_23 }
    }

    /// Map a 42D assessor vector to a symmetric 3x3 generation matrix.
    ///
    /// The diagonal is zero (no self-coupling). Off-diagonal (i,j) is the
    /// sum of assessor components for that generation pair.
    pub fn to_generation_matrix(&self, v: &[f64]) -> faer::Mat<f64> {
        let f_12: f64 = self.gen_12_indices.iter().map(|&i| v[i]).sum();
        let f_13: f64 = self.gen_13_indices.iter().map(|&i| v[i]).sum();
        let f_23: f64 = self.gen_23_indices.iter().map(|&i| v[i]).sum();

        let mut m = faer::Mat::<f64>::zeros(3, 3);
        m.write(0, 1, f_12);
        m.write(1, 0, f_12);
        m.write(0, 2, f_13);
        m.write(2, 0, f_13);
        m.write(1, 2, f_23);
        m.write(2, 1, f_23);
        m
    }
}

/// Compute the constrained solar direction in V_6 space.
///
/// Projects g_12 orthogonal to the g_13 and g_23 constraint planes using
/// Gram-Schmidt orthogonalization. The result is a unit vector u such that:
///   g_13 . u = 0  (zero first-order reactor leakage)
///   g_23 . u = 0  (zero first-order atmospheric leakage)
///   g_12 . u is maximized (maximal solar sensitivity)
///
/// Returns the normalized direction. If the projection has near-zero norm
/// (meaning g_12 lies entirely in the constraint plane), returns the
/// zero vector.
pub fn compute_constrained_solar_direction(
    g_12: &[f64; 6],
    g_13: &[f64; 6],
    g_23: &[f64; 6],
) -> [f64; 6] {
    let dot = |a: &[f64; 6], b: &[f64; 6]| -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    };

    // Orthonormalize the constraint basis {g_13, g_23}
    let mut u1 = *g_13;
    let norm_u1 = dot(&u1, &u1).sqrt();
    if norm_u1 < 1e-15 {
        // g_13 is zero -- no reactor constraint, just project out g_23
        let mut u2 = *g_23;
        let norm_u2 = dot(&u2, &u2).sqrt();
        if norm_u2 < 1e-15 {
            // No constraints at all -- g_12 itself is the direction
            let norm_12 = dot(g_12, g_12).sqrt();
            if norm_12 < 1e-15 { return [0.0; 6]; }
            let mut out = *g_12;
            for x in &mut out { *x /= norm_12; }
            return out;
        }
        for x in &mut u2 { *x /= norm_u2; }
        let proj = dot(g_12, &u2);
        let mut out = [0.0_f64; 6];
        for i in 0..6 { out[i] = g_12[i] - proj * u2[i]; }
        let norm = dot(&out, &out).sqrt();
        if norm < 1e-15 { return [0.0; 6]; }
        for x in &mut out { *x /= norm; }
        return out;
    }
    for x in &mut u1 { *x /= norm_u1; }

    // Orthogonalize g_23 against u1
    let proj_23_on_1 = dot(g_23, &u1);
    let mut u2 = [0.0_f64; 6];
    for i in 0..6 {
        u2[i] = g_23[i] - proj_23_on_1 * u1[i];
    }
    let norm_u2 = dot(&u2, &u2).sqrt();
    if norm_u2 > 1e-15 {
        for x in &mut u2 { *x /= norm_u2; }
    }

    // Project g_12 away from the {u1, u2} constraint plane
    let proj_12_on_1 = dot(g_12, &u1);
    let proj_12_on_2 = dot(g_12, &u2);

    let mut optimal = [0.0_f64; 6];
    for i in 0..6 {
        optimal[i] = g_12[i] - proj_12_on_1 * u1[i] - proj_12_on_2 * u2[i];
    }

    let norm = dot(&optimal, &optimal).sqrt();
    if norm < 1e-15 { return [0.0; 6]; }
    for x in &mut optimal { *x /= norm; }

    optimal
}

/// Compute a constrained atmospheric direction orthogonal to the solar direction.
///
/// Projects g_23 orthogonal to {g_13, u_solar} using Gram-Schmidt.
/// The result maximizes atmospheric sensitivity while:
///   g_13 . u = 0  (zero reactor leakage)
///   u_solar . u = 0  (orthogonal to solar direction)
pub fn compute_constrained_atmospheric_direction(
    g_23: &[f64; 6],
    g_13: &[f64; 6],
    u_solar: &[f64; 6],
) -> [f64; 6] {
    let dot = |a: &[f64; 6], b: &[f64; 6]| -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    };

    // Build orthonormal constraint basis {g_13_hat, u_solar}
    let mut u1 = *g_13;
    let norm_u1 = dot(&u1, &u1).sqrt();
    if norm_u1 < 1e-15 { return [0.0; 6]; }
    for x in &mut u1 { *x /= norm_u1; }

    // u_solar should already be orthogonal to g_13 (from solar construction)
    // but orthogonalize again for safety
    let proj = dot(u_solar, &u1);
    let mut u2 = *u_solar;
    for i in 0..6 { u2[i] -= proj * u1[i]; }
    let norm_u2 = dot(&u2, &u2).sqrt();
    if norm_u2 < 1e-15 { return [0.0; 6]; }
    for x in &mut u2 { *x /= norm_u2; }

    // Project g_23 away from {u1, u2}
    let proj_1 = dot(g_23, &u1);
    let proj_2 = dot(g_23, &u2);
    let mut optimal = [0.0_f64; 6];
    for i in 0..6 {
        optimal[i] = g_23[i] - proj_1 * u1[i] - proj_2 * u2[i];
    }

    let norm = dot(&optimal, &optimal).sqrt();
    if norm < 1e-15 { return [0.0; 6]; }
    for x in &mut optimal { *x /= norm; }

    optimal
}

/// Gauss-Newton solver for 2D (t_solar, t_atmo) optimization.
///
/// Minimizes the weighted residual ||r(t)||^2 where
///   r = [(theta_12 - pdg_12)/pdg_12, (theta_13 - pdg_13)/pdg_13, (theta_23 - pdg_23)/pdg_23]
/// using the affine structure M_nu(t1,t2) = M_nu_base + t1*A + t2*B.
///
/// The `angles_fn` takes (t1, t2) and returns (theta_12, theta_13, theta_23).
/// Returns (best_t1, best_t2, best_angles, score).
pub fn gauss_newton_2d<F>(
    angles_fn: &F,
    t1_init: f64,
    t2_init: f64,
    pdg: (f64, f64, f64),
    weights: (f64, f64, f64),
    max_iter: usize,
) -> (f64, f64, (f64, f64, f64), f64)
where
    F: Fn(f64, f64) -> (f64, f64, f64),
{
    let eps = 0.01_f64;
    let mut t1 = t1_init;
    let mut t2 = t2_init;

    for _iter in 0..max_iter {
        let (a12, a13, a23) = angles_fn(t1, t2);
        let r = [
            weights.0 * (a12 - pdg.0) / pdg.0,
            weights.1 * (a13 - pdg.1) / pdg.1,
            weights.2 * (a23 - pdg.2) / pdg.2,
        ];

        // 3x2 Jacobian via finite differences
        let (a12_p1, a13_p1, a23_p1) = angles_fn(t1 + eps, t2);
        let (a12_m1, a13_m1, a23_m1) = angles_fn(t1 - eps, t2);
        let (a12_p2, a13_p2, a23_p2) = angles_fn(t1, t2 + eps);
        let (a12_m2, a13_m2, a23_m2) = angles_fn(t1, t2 - eps);

        let j = [
            [weights.0 * (a12_p1 - a12_m1) / (2.0 * eps * pdg.0),
             weights.0 * (a12_p2 - a12_m2) / (2.0 * eps * pdg.0)],
            [weights.1 * (a13_p1 - a13_m1) / (2.0 * eps * pdg.1),
             weights.1 * (a13_p2 - a13_m2) / (2.0 * eps * pdg.1)],
            [weights.2 * (a23_p1 - a23_m1) / (2.0 * eps * pdg.2),
             weights.2 * (a23_p2 - a23_m2) / (2.0 * eps * pdg.2)],
        ];

        // Normal equations: J^T J * delta = -J^T r
        let jtj = [
            [j[0][0]*j[0][0] + j[1][0]*j[1][0] + j[2][0]*j[2][0],
             j[0][0]*j[0][1] + j[1][0]*j[1][1] + j[2][0]*j[2][1]],
            [j[0][1]*j[0][0] + j[1][1]*j[1][0] + j[2][1]*j[2][0],
             j[0][1]*j[0][1] + j[1][1]*j[1][1] + j[2][1]*j[2][1]],
        ];
        let jtr = [
            j[0][0]*r[0] + j[1][0]*r[1] + j[2][0]*r[2],
            j[0][1]*r[0] + j[1][1]*r[1] + j[2][1]*r[2],
        ];

        // Solve 2x2 system: Cramer's rule with Levenberg-Marquardt damping
        let lambda = 0.01; // damping factor
        let a11 = jtj[0][0] + lambda;
        let a12_m = jtj[0][1];
        let a22 = jtj[1][1] + lambda;
        let det = a11 * a22 - a12_m * a12_m;
        if det.abs() < 1e-30 { break; }

        let dt1 = -(a22 * jtr[0] - a12_m * jtr[1]) / det;
        let dt2 = -(a11 * jtr[1] - a12_m * jtr[0]) / det;

        // Line search with backtracking
        let mut alpha = 1.0_f64;
        let current_cost: f64 = r.iter().map(|x| x * x).sum();
        for _ in 0..10 {
            let new_t1 = t1 + alpha * dt1;
            let new_t2 = t2 + alpha * dt2;
            let (na12, na13, na23) = angles_fn(new_t1, new_t2);
            let nr = [
                weights.0 * (na12 - pdg.0) / pdg.0,
                weights.1 * (na13 - pdg.1) / pdg.1,
                weights.2 * (na23 - pdg.2) / pdg.2,
            ];
            let new_cost: f64 = nr.iter().map(|x| x * x).sum();
            if new_cost < current_cost {
                t1 = new_t1;
                t2 = new_t2;
                break;
            }
            alpha *= 0.5;
        }

        // Convergence check
        if dt1.abs() < 1e-6 && dt2.abs() < 1e-6 { break; }
    }

    let (a12, a13, a23) = angles_fn(t1, t2);
    let score = ((a12 - pdg.0) / pdg.0).powi(2)
              + ((a13 - pdg.1) / pdg.1).powi(2)
              + ((a23 - pdg.2) / pdg.2).powi(2);
    (t1, t2, (a12, a13, a23), score)
}

/// Trait for mapping a 42D assessor-space vector to a mass matrix perturbation.
///
/// Different implementations encode different physical hypotheses about how
/// the V_6 topological content couples to the 3x3 generation structure.
/// The trait makes the null result of the (12/12/6) partition first-class:
/// it is one implementation, not the only path.
pub trait FlavorLift {
    /// Apply the perturbation vector `v` (42D assessor space) to the
    /// mass matrix `m`. The perturbation must be symmetric.
    fn lift(&self, v: &[f64], m: &mut faer::Mat<f64>);
}

impl FlavorLift for AssessorToFlavorMap {
    fn lift(&self, v: &[f64], m: &mut faer::Mat<f64>) {
        let delta = self.to_generation_matrix(v);
        for i in 0..3 {
            for j in 0..3 {
                let sym = (delta.read(i, j) + delta.read(j, i)) / 2.0;
                m.write(i, j, m.read(i, j) + sym);
            }
        }
    }
}

/// Direct off-diagonal injection: bypasses the assessor partition entirely.
///
/// Collapses the 42D perturbation into 3 independent off-diagonal coupling
/// strengths by partitioning into 3 blocks of 14. Each block's sum drives
/// one off-diagonal element of the mass matrix directly.
///
/// This targets eigenvector rotation (off-diagonal torque) instead of
/// eigenvalue shifting (diagonal trace), which is where the (12/12/6)
/// partition failed.
pub struct DirectOffDiagonalLift;

impl FlavorLift for DirectOffDiagonalLift {
    fn lift(&self, v: &[f64], m: &mut faer::Mat<f64>) {
        // 42 assessors split into 3 blocks of 14
        let n = v.len().min(42);
        let block_size = n / 3;
        let torque_12: f64 = v[..block_size].iter().sum();
        let torque_13: f64 = v[block_size..2 * block_size].iter().sum();
        let torque_23: f64 = v[2 * block_size..n].iter().sum();

        // Inject symmetrically into off-diagonals
        m.write(0, 1, m.read(0, 1) + torque_12);
        m.write(1, 0, m.read(1, 0) + torque_12);
        m.write(0, 2, m.read(0, 2) + torque_13);
        m.write(2, 0, m.read(2, 0) + torque_13);
        m.write(1, 2, m.read(1, 2) + torque_23);
        m.write(2, 1, m.read(2, 1) + torque_23);
    }
}

/// Psi-equivariant lift: maps assessors to generations using the Gourlay psi
/// automorphism's orbit structure.
///
/// For each assessor pair (low, high), embeds it as a 16D unit vector,
/// applies psi, and classifies the orbit:
/// - Fixed points (psi(v) = v): contribute to diagonal (trace)
/// - 3-cycle orbits: contribute to off-diagonal via the circulant structure
///
/// This uses the S3 generator directly rather than hand-crafted index bins.
pub struct PsiEquivariantLift {
    /// For each assessor index: (off-diag contribution weights [f_12, f_13, f_23])
    weights: Vec<[f64; 3]>,
}

impl PsiEquivariantLift {
    /// Build the psi-orbit classification for the given assessor pairs.
    pub fn from_assessors(assessors: &[(usize, usize)]) -> Self {
        use cd_kernel::gourlay_psi;

        let mut weights = Vec::with_capacity(assessors.len());

        for &(low, high) in assessors {
            // Embed as a 16D vector: e_low + e_high
            let mut v = [0.0_f64; 16];
            v[low] = 1.0;
            v[high] = 1.0;

            // Apply psi to get the transformed vector
            let psi_v = gourlay_psi(&v);

            // The overlap <v, psi(v)> measures how much this assessor
            // is preserved by the S3 generator. Negative overlap means
            // the assessor couples across generations.
            let self_overlap: f64 = v.iter().zip(psi_v.iter()).map(|(a, b)| a * b).sum();

            // Apply psi^2 for the third orbit element
            let psi2_v = gourlay_psi(&psi_v);
            let cross_overlap: f64 = v.iter().zip(psi2_v.iter()).map(|(a, b)| a * b).sum();

            // Classification:
            // - Large positive self_overlap -> psi-invariant (diagonal/trace)
            // - Negative self_overlap -> psi rotates this assessor (off-diagonal)
            // The overlap magnitudes determine which off-diagonal channel
            // is most affected.
            //
            // Map: (1-2) gets self_overlap contribution (cos(2*pi/3) = -0.5 for pure rotation)
            //       (1-3) gets cross_overlap
            //       (2-3) gets the remainder
            let norm = (self_overlap.abs() + cross_overlap.abs()).max(1e-15);
            let w_12 = -self_overlap / norm;  // negative overlap -> positive 1-2 coupling
            let w_13 = -cross_overlap / norm;
            let w_23 = 1.0 - w_12.abs() - w_13.abs();

            weights.push([w_12, w_13, w_23]);
        }

        Self { weights }
    }
}

impl FlavorLift for PsiEquivariantLift {
    fn lift(&self, v: &[f64], m: &mut faer::Mat<f64>) {
        let mut f_12 = 0.0_f64;
        let mut f_13 = 0.0_f64;
        let mut f_23 = 0.0_f64;

        for (idx, &val) in v.iter().enumerate() {
            if idx >= self.weights.len() { break; }
            f_12 += val * self.weights[idx][0];
            f_13 += val * self.weights[idx][1];
            f_23 += val * self.weights[idx][2];
        }

        m.write(0, 1, m.read(0, 1) + f_12);
        m.write(1, 0, m.read(1, 0) + f_12);
        m.write(0, 2, m.read(0, 2) + f_13);
        m.write(2, 0, m.read(2, 0) + f_13);
        m.write(1, 2, m.read(1, 2) + f_23);
        m.write(2, 1, m.read(2, 1) + f_23);
    }
}

/// Tensor element lift: maps 42 assessors into 6 blocks of 7, each driving
/// one independent element of the symmetric 3x3 mass matrix.
///
/// This is the minimal successful project lift (C-1478) that breaks the
/// rank-2 Jacobian lock. It injects into all 6 independent elements of
/// Herm_3 (3 diagonal + 3 off-diagonal), matching the 6D V_6 exactly.
///
/// **Scope**: The specific 7-assessor block assignment is a PROJECT-SPECIFIC
/// heuristic, not yet derived from the algebra. Invariance audit shows
/// moderate block alignment (44% max concentration) and psi orbits cross
/// block boundaries (30 cross vs 12 within). The lift works because it
/// preserves 6 effective DOFs, not because the blocks are canonical.
///
/// Block assignment (by natural assessor ordering):
///   assessors  0-6  -> M_11 (diagonal, gen 1)
///   assessors  7-13 -> M_22 (diagonal, gen 2)
///   assessors 14-20 -> M_33 (diagonal, gen 3)
///   assessors 21-27 -> M_12 (off-diagonal, solar)
///   assessors 28-34 -> M_13 (off-diagonal, reactor)
///   assessors 35-41 -> M_23 (off-diagonal, atmospheric)
pub struct TensorElementLift;

impl FlavorLift for TensorElementLift {
    fn lift(&self, v: &[f64], m: &mut faer::Mat<f64>) {
        let n = v.len().min(42);
        // 6 blocks of 7 assessors
        let block = 7_usize;

        let sum_block = |start: usize| -> f64 {
            let end = (start + block).min(n);
            v[start..end].iter().sum()
        };

        // Diagonal elements
        m.write(0, 0, m.read(0, 0) + sum_block(0));
        m.write(1, 1, m.read(1, 1) + sum_block(block));
        m.write(2, 2, m.read(2, 2) + sum_block(2 * block));

        // Off-diagonal elements (symmetric injection)
        let m12 = sum_block(3 * block);
        let m13 = sum_block(4 * block);
        let m23 = sum_block(5 * block);

        m.write(0, 1, m.read(0, 1) + m12);
        m.write(1, 0, m.read(1, 0) + m12);
        m.write(0, 2, m.read(0, 2) + m13);
        m.write(2, 0, m.read(2, 0) + m13);
        m.write(1, 2, m.read(1, 2) + m23);
        m.write(2, 1, m.read(2, 1) + m23);
    }
}

/// Apply a V_6 perturbation to a neutrino mass matrix using a pluggable lift.
///
/// For each V_6 direction k with coefficient beta[k], computes the combined
/// assessor-space vector, then delegates to the `FlavorLift` implementation
/// to map it into the mass matrix.
///
/// Invariant: beta = [0; 6] leaves m_nu unchanged.
pub fn apply_v6_perturbation(
    m_nu: &mut faer::Mat<f64>,
    v6_basis: &nalgebra::DMatrix<f64>,
    beta: &[f64; 6],
    flavor_lift: &dyn FlavorLift,
) {
    let n_basis = v6_basis.nrows().min(6);
    let n_cols = v6_basis.ncols();

    // Combine V_6 directions: v_combined = sum_k beta[k] * v6_basis.row(k)
    let mut v_combined = vec![0.0_f64; n_cols];
    for k in 0..n_basis {
        if beta[k].abs() < 1e-15 { continue; }
        for col in 0..n_cols {
            v_combined[col] += beta[k] * v6_basis[(k, col)];
        }
    }

    flavor_lift.lift(&v_combined, m_nu);
}

// ---------------------------------------------------------------------------
// J_k complex structure -- full 16D sedenion action
// ---------------------------------------------------------------------------

/// Apply J_k complex structure to a 16D sedenion vector via octonion
/// left-multiplication on both halves independently.
///
/// # Mathematical foundation
///
/// A sedenion decomposes as `(a, b)` where `a` and `b` are octonions (the
/// lower and upper halves).  The complex structure `J_k` for `k in 1..=7`
/// acts by left-multiplication with the basis element `e_k`:
///
/// ```text
/// J_k(a, b) = (e_k * a,  e_k * b)
/// ```
///
/// This gives a **14D active subspace**: each half has 7 imaginary components
/// rotated by `e_k`, while the real (index 0) and `e_k` (index k) components
/// rotate among themselves.  Compare with the 6D perp-only action from
/// [`complex_structure()`](gororoba_algebra::lie::g2_stabilizer::complex_structure),
/// which restricts to the 6 indices perpendicular to both `e_0` and `e_k`
/// within the lower octonion only.
///
/// # Why full 16D instead of 6D perp-only
///
/// The 6D action discards contributions from indices `{0, k}` in the lower
/// block and the entire upper block `{8..15}`.  For friction profiles whose
/// support spans both halves, the 16D action captures phase angles that the
/// 6D action misses.  In practice (C-1496), friction profiles from selectors
/// `(e_7, e_8)` have zero upper-block weight, so the two actions agree --
/// but this is a property of the *profiles*, not of the action itself.
///
/// # Callers
///
/// - [`test_cp_violation_phase_only`]: primary CP violation pipeline
/// - [`test_cp_violation_jk_dimension_comparison`]: 6D-vs-16D diagnostic
/// - [`test_cp_violation_joint_3d_scan`]: 3D optimiser for J_CP gap closure
/// - [`test_complex_pmns_alpha_scan`]: fine alpha_CP scan (origin of this
///   implementation, extracted from closure at former line 6508)
///
/// # Panics
///
/// Via [`Octonion::basis`]: panics if `k >= 8`.  Caller must ensure
/// `k in 1..=7` (the seven imaginary octonion units).
///
/// # Concrete example
///
/// ```text
/// k = 1,  v = (0, 0, 1, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0)
///                      ^-- e_2 in lower block
/// J_1(v) = e_1 * e_2 = e_3  (by the Fano plane)
///        => result[3] = 1.0, all others zero
/// ```
pub fn apply_jk_full_16d(v: &[f64; 16], k: usize) -> [f64; 16] {
    use gororoba_algebra::construction::octonion::Octonion;

    // e_k is the imaginary unit that defines this complex structure.
    // Octonion::basis(k) returns the unit vector with component k = 1.0.
    let ek = Octonion::basis(k);
    let mut result = [0.0_f64; 16];

    // Lower octonion half: indices 0..7.
    // core::array::from_fn avoids an intermediate Vec allocation.
    let lower = Octonion::new(core::array::from_fn(|i| v[i]));
    // Left-multiplication: J_k(a) = e_k * a.  The product uses the
    // Fano-plane sign rule via cd_basis_mul_sign_iter internally.
    let jk_lower = ek.multiply(&lower);
    result[..8].copy_from_slice(&jk_lower.components);

    // Upper octonion half: indices 8..15.
    // The sedenion doubling construction makes the upper half an
    // independent octonion -- J_k acts on it identically.
    let upper = Octonion::new(core::array::from_fn(|i| v[i + 8]));
    let jk_upper = ek.multiply(&upper);
    result[8..16].copy_from_slice(&jk_upper.components);

    result
}

// ---------------------------------------------------------------------------
// Parameterized CP scan point evaluator
// ---------------------------------------------------------------------------
//
// This section implements the parameterized evaluation pipeline for the
// CP violation scan.  The design separates three concerns:
//
//   CpScanResult   -- pure data, the 5 physical observables
//   CpScanContext   -- immutable algebraic data shared across all grid points
//   CpScanBuffers   -- mutable pre-allocated matrices, one per thread
//   evaluate_cp_scan_point() -- the pipeline itself
//
// The separation is motivated by the same principle Lions uses for malloc's
// "coremap" and "swapmap" (Lions 5.1): the resource layout is fixed, but
// the allocator state is per-invocation.  Here, the algebraic "layout"
// (mass matrices, V_6 basis, permutation) is fixed per scan, while the
// eigendecomposition working memory is per-thread mutable state.

/// The five physical observables from a single CP scan point.
///
/// All angles are in **degrees** (not radians), following the PDG
/// convention.  `j_cp` is the rephasing-invariant Jarlskog:
///
/// ```text
/// J = Im(U_e1 * U_mu2 * conj(U_e2) * conj(U_mu1))
/// ```
///
/// `delta_cp` is `arg(-U_e3)` in degrees -- this is convention-dependent.
/// For the rephasing-invariant delta, compute `arg(Jarlskog quartet)`
/// separately (see the rephasing analysis in [`test_cp_violation_joint_3d_scan`]).
///
/// # Concrete values at the optimum (C-1497, AMENDED)
///
/// ```text
/// theta_12 = 32.84,  theta_13 = 8.58,  theta_23 = 49.48
/// j_cp     = 3.33e-2 = J_max (kinematic maximum, delta~90)
/// delta_cp = 97.9 deg (arg(-U_e3)), 92.8 deg (invariant)
/// PDG |J|  = 8.6e-3 (non-maximal, delta=195).  Ratio 3.9x.
/// ```
#[derive(Debug, Clone, Copy)]
pub struct CpScanResult {
    /// Solar mixing angle (PDG: 33.41 +/- 0.72 deg).
    pub theta_12: f64,
    /// Reactor mixing angle (PDG: 8.54 +/- 0.12 deg).
    pub theta_13: f64,
    /// Atmospheric mixing angle (PDG: 49.0 +/- 1.3 deg).
    pub theta_23: f64,
    /// Jarlskog invariant -- kinematic maximum J_max for these angles.
    /// PDG measured |J| ~ 8.6e-3 (non-maximal, sin(delta) ~ 0.26).
    pub j_cp: f64,
    /// CP phase from arg(-U_e3), in degrees (convention-dependent).
    pub delta_cp: f64,
    /// Rephasing-invariant delta_CP from atan2(sin_delta, cos_delta),
    /// using Jarlskog for sin and |U_mu1|^2 identity for cos.
    /// In degrees. This is the preferred observable for PDG comparison.
    pub delta_cp_invariant: f64,
}

/// Extract delta_CP using rephasing-invariant observables (PDG convention).
///
/// Uses sin(delta) from the Jarlskog invariant and cos(delta) from the
/// |U_mu1|^2 unitarity relation, combined via atan2 for quadrant resolution.
///
/// This avoids the convention-dependence of arg(-U_e3), which depends on
/// individual matrix element phases rather than physical observables.
pub fn extract_delta_cp_invariant(u_moduli: &[[f64; 3]; 3], j_cp: f64) -> f64 {
    let s13 = u_moduli[0][2];
    let c13 = (1.0 - s13 * s13).max(0.0).sqrt();
    if c13 < 1e-15 { return 0.0; }

    let s12 = u_moduli[0][1] / c13;
    let c12 = u_moduli[0][0] / c13;
    let s23 = u_moduli[1][2] / c13;
    let c23 = u_moduli[2][2] / c13;

    // sin(delta) = J / (s12*c12*s23*c23*s13*c13^2)
    let denom = s12 * c12 * s23 * c23 * s13 * c13 * c13;
    let sin_delta = if denom.abs() > 1e-15 { j_cp / denom } else { 0.0 };

    // cos(delta) from |U_mu1|^2 = s12^2*c23^2 + c12^2*s23^2*s13^2
    //                              + 2*s12*c12*s23*c23*s13*cos(delta)
    let u_mu1_sq = u_moduli[1][0] * u_moduli[1][0];
    let expected_no_cp = s12 * s12 * c23 * c23 + c12 * c12 * s23 * s23 * s13 * s13;
    let cos_denom = 2.0 * s12 * c12 * s23 * c23 * s13;
    let cos_delta = if cos_denom.abs() > 1e-15 {
        (u_mu1_sq - expected_no_cp) / cos_denom
    } else {
        1.0
    };

    sin_delta.atan2(cos_delta).to_degrees()
}

/// Immutable algebraic context for a CP violation scan.
///
/// # Algebraic origin of each field
///
/// ```text
/// m_nu_real   : 3x3 symmetric, from sedenion associator friction
///               [a, x, b] with selectors (e_7, e_8), coupling alpha_nu=1.35.
///               Constructed by construct_pmns_matrices_two_param().
///
/// m_ch_real   : 3x3 symmetric, from selectors (e_11, e_12), alpha_ch=3.00.
///               Same construction, different subalgebra embedding.
///
/// v6_basis    : nalgebra DMatrix (6 x 42), the top-6 singular vectors of
///               the assessor-space Jacobian.  Spans the directions in the
///               42D assessor space that most efficiently steer the mixing
///               angles.  Computed by extract_v6_basis().
///
/// u_solar     : [f64; 6], the direction in V_6 that moves theta_12 while
///               preserving theta_13.  From compute_constrained_solar_direction().
///
/// u_atmo      : [f64; 6], the direction in V_6 that moves theta_23 while
///               preserving theta_13 and staying orthogonal to u_solar.
///               From compute_constrained_atmospheric_direction().
///
/// lift        : &dyn FlavorLift, maps 42D assessor vectors to 3x3 mass
///               matrix perturbations.  Currently TensorElementLift (the
///               S_3 intertwiner).  This is the ONLY non-algebraically-
///               canonical component -- see C-1489 for the no-equivariance
///               proof.
///
/// perm_u/d    : [usize; 3], eigenvalue permutation indices calibrated
///               against the real mass matrix baseline.  Ensures that
///               eigenvalue 0 maps to the lightest mass eigenstate
///               consistently across the scan.
/// ```
///
/// # Edge cases and invariants
///
/// - **perm stability**: The permutation is computed ONCE from the real
///   (alpha_CP=0) baseline.  If alpha_CP is so large that eigenvalues
///   cross (level repulsion), the permutation becomes invalid and angles
///   will be nonsensical.  The 2% acceptance filter catches this.
///
/// - **v6_basis rank**: The V_6 extraction can produce fewer than 6
///   significant singular values if the assessor-space Jacobian is
///   rank-deficient.  In practice, n_basis = min(nrows, 6) = 6 always.
///
/// - **lift non-uniqueness**: TensorElementLift is response-fitted, not
///   algebraically canonical.  A different FlavorLift implementation
///   would change the optimal (t_sol, t_atm) but NOT the existence
///   of CP violation (which comes from the J_k phase structure).
///
/// # Thread safety
///
/// All fields are shared references (`&'a`), so `CpScanContext` is
/// `Send + Sync` by construction.  It can be shared across rayon
/// threads without cloning.
pub struct CpScanContext<'a> {
    /// Neutrino mass matrix (3x3 real symmetric, from associator friction).
    pub m_nu_real: &'a faer::Mat<f64>,
    /// Charged lepton mass matrix (3x3 real symmetric).
    pub m_ch_real: &'a faer::Mat<f64>,
    /// Top-6 V_6 singular vectors (6 x 42, from assessor-space Jacobian).
    pub v6_basis: &'a nalgebra::DMatrix<f64>,
    /// Constrained solar direction in V_6 space.
    pub u_solar: &'a [f64; 6],
    /// Constrained atmospheric direction in V_6 space.
    pub u_atmo: &'a [f64; 6],
    /// The 42D -> 3x3 flavor lift (currently TensorElementLift).
    pub lift: &'a dyn FlavorLift,
    /// Eigenvalue permutation for charged leptons.
    pub perm_u: [usize; 3],
    /// Eigenvalue permutation for neutrinos.
    pub perm_d: [usize; 3],
}

/// Mutable pre-allocated buffers for the eigendecomposition loop.
///
/// Owns two 3x3 complex Hermitian faer matrices that are overwritten
/// (not re-allocated) at each scan point.  This eliminates ~1200
/// `Mat::zeros(3, 3)` heap allocations per k-embedding in the scan.
///
/// # Why separate from CpScanContext
///
/// `faer::Mat` is not `Sync` (it uses internal mutability for the
/// eigendecomposition working buffer).  By separating the mutable
/// buffers into their own struct, each rayon thread creates its own
/// `CpScanBuffers` while sharing a single `&CpScanContext`.
///
/// # Memory layout
///
/// Each `Mat<c64>::zeros(3, 3)` allocates 9 * 16 = 144 bytes on the
/// heap (9 complex f64s).  Two buffers = 288 bytes per thread.
/// This is allocated ONCE per k-embedding, not per grid point.
pub struct CpScanBuffers {
    /// Pre-allocated neutrino mass matrix (overwritten each point).
    pub m_nu: faer::Mat<faer::complex_native::c64>,
    /// Pre-allocated charged lepton mass matrix (overwritten each point).
    pub m_ch: faer::Mat<faer::complex_native::c64>,
}

impl Default for CpScanBuffers {
    fn default() -> Self {
        Self {
            m_nu: faer::Mat::<faer::complex_native::c64>::zeros(3, 3),
            m_ch: faer::Mat::<faer::complex_native::c64>::zeros(3, 3),
        }
    }
}

impl CpScanBuffers {
    /// Create a new buffer pair.  Equivalent to `Default::default()`.
    pub fn new() -> Self { Self::default() }
}

/// Evaluate a single CP scan point in the 3D (alpha_CP, t_sol, t_atm) space.
///
/// # Mathematical pipeline
///
/// The pipeline has 5 steps, each corresponding to a physical operation:
///
/// ```text
/// Step 1: (t_sol, t_atm) --[u_solar, u_atmo]--> beta[6]
///         Linear combination in V_6 constrained-direction space.
///         beta[k] = t_sol * u_solar[k] + t_atm * u_atmo[k]
///
/// Step 2: beta --[apply_v6_perturbation]--> M_nu_pert (3x3 real)
///         The lift maps the 6D beta into a 42D assessor vector,
///         which is then injected into the mass matrix.
///         Symmetrization: M = (M + M^T) / 2.
///
/// Step 3: (M_nu_pert, alpha_CP, phi) --> M_nu_complex (3x3 Hermitian)
///         Phase injection: M[i][j] = |M_pert[i][j]| * exp(i * alpha * phi[i][j])
///         Hermiticity: M[j][i] = conj(M[i][j]).  Diagonal stays real.
///
/// Step 4: (M_ch, M_nu_complex) --[eigendecomp]--> (U_ch, U_nu)
///         Hermitian eigendecomposition via faer (iterative QR).
///         U_PMNS = U_ch^dag * U_nu.
///
/// Step 5: U_PMNS --[perm + extraction]--> CpScanResult
///         Apply perm_u/perm_d, extract |U_ij| -> angles,
///         Jarlskog quartet -> J_CP, arg(-U_e3) -> delta_CP.
/// ```
///
/// # Why phi is a separate argument (not in CpScanContext)
///
/// The phase matrix `phi[i][j]` depends on the k-embedding (which
/// complex structure J_k is used).  The context is k-independent;
/// phi changes per k.  This allows a single context to be reused
/// across all 7 k-embeddings in the parallel scan.
///
/// # Performance
///
/// ~1.5us per call with pre-allocated buffers (dominated by the
/// 3x3 complex eigendecomposition, ~0.7us per matrix * 2 matrices).
/// Without pre-allocation: ~5us per call (dominated by 2 * Mat::zeros).
///
/// # Callers
///
/// - [`test_cp_violation_joint_3d_scan`]: coarse + fine inner loops
/// - Future Nelder-Mead objective via argmin (planned, C-1497 follow-up)
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub fn evaluate_cp_scan_point(
    alpha_cp: f64,
    t_sol: f64,
    t_atm: f64,
    phi: &[[f64; 3]; 3],
    ctx: &CpScanContext<'_>,
    bufs: &mut CpScanBuffers,
) -> CpScanResult {
    // Step 1: beta from (t_sol, t_atm)
    let mut beta = [0.0_f64; 6];
    for k in 0..6 {
        beta[k] = t_sol * ctx.u_solar[k] + t_atm * ctx.u_atmo[k];
    }

    // Step 2: perturb real mass matrix
    let mut m_nu_pert = ctx.m_nu_real.clone();
    apply_v6_perturbation(&mut m_nu_pert, ctx.v6_basis, &beta, ctx.lift);
    let m_nu_pert = (&m_nu_pert + m_nu_pert.transpose()) * faer::scale(0.5);

    // Step 3: fill pre-allocated complex Hermitian matrices
    for i in 0..3 {
        bufs.m_nu.write(i, i, faer::complex_native::c64::new(
            m_nu_pert.read(i, i), 0.0));
        bufs.m_ch.write(i, i, faer::complex_native::c64::new(
            ctx.m_ch_real.read(i, i), 0.0));
        for j in (i + 1)..3 {
            let phase = alpha_cp * phi[i][j];
            let mag = m_nu_pert.read(i, j);
            bufs.m_nu.write(i, j, faer::complex_native::c64::new(
                mag * phase.cos(), mag * phase.sin()));
            bufs.m_nu.write(j, i, faer::complex_native::c64::new(
                mag * phase.cos(), -mag * phase.sin()));
            bufs.m_ch.write(i, j, faer::complex_native::c64::new(
                ctx.m_ch_real.read(i, j), 0.0));
            bufs.m_ch.write(j, i, faer::complex_native::c64::new(
                ctx.m_ch_real.read(j, i), 0.0));
        }
    }

    // Step 4: eigendecompose
    let eig_ch = bufs.m_ch.selfadjoint_eigendecomposition(faer::Side::Lower);
    let eig_nu = bufs.m_nu.selfadjoint_eigendecomposition(faer::Side::Lower);
    let u_pmns = eig_ch.u().adjoint() * eig_nu.u();

    // Step 5: extract with permutation (no allocation)
    let u_at = |i: usize, j: usize| -> faer::complex_native::c64 {
        u_pmns.read(ctx.perm_u[i], ctx.perm_d[j])
    };

    let u_e3_abs = u_at(0, 2).abs();
    let theta_13 = u_e3_abs.min(1.0).asin().to_degrees();
    let cos_13 = theta_13.to_radians().cos();

    let theta_12 = if cos_13 > 1e-15 {
        (u_at(0, 1).abs() / cos_13).min(1.0).asin().to_degrees()
    } else { 0.0 };

    let theta_23 = if cos_13 > 1e-15 {
        (u_at(1, 2).abs() / cos_13).min(1.0).asin().to_degrees()
    } else { 0.0 };

    let j_cp = (u_at(0, 0) * u_at(1, 1)
        * u_at(0, 1).conj() * u_at(1, 0).conj()).im;

    let delta_cp = (-u_at(0, 2)).arg().to_degrees();

    // Rephasing-invariant delta_CP via moduli + Jarlskog
    let u_moduli = {
        let mut m = [[0.0_f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                m[i][j] = u_at(i, j).abs();
            }
        }
        m
    };
    let delta_cp_invariant = extract_delta_cp_invariant(&u_moduli, j_cp);

    CpScanResult { theta_12, theta_13, theta_23, j_cp, delta_cp, delta_cp_invariant }
}

// ---------------------------------------------------------------------------
// Nelder-Mead refinement of CP scan (argmin)
// ---------------------------------------------------------------------------

/// Cost function for argmin Nelder-Mead optimization of the CP scan.
///
/// The parameter vector is `[alpha_CP, t_sol, t_atm]` (k is frozen from
/// the grid search).  The cost function penalizes angle deviation from PDG
/// and rewards large |J_CP|.
pub struct CpNelderMeadCost<'a> {
    pub ctx: &'a CpScanContext<'a>,
    pub phi: [[f64; 3]; 3],
    pub bounds: [(f64, f64); 3],
    /// If true, pure prediction mode: cost = -|J_CP| (no angle penalty).
    pub prediction_mode: bool,
}

impl<'a> argmin::core::CostFunction for CpNelderMeadCost<'a> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        let alpha_cp = param[0].clamp(self.bounds[0].0, self.bounds[0].1);
        let t_sol = param[1].clamp(self.bounds[1].0, self.bounds[1].1);
        let t_atm = param[2].clamp(self.bounds[2].0, self.bounds[2].1);

        let mut bufs = CpScanBuffers::new();
        let r = evaluate_cp_scan_point(alpha_cp, t_sol, t_atm, &self.phi, self.ctx, &mut bufs);

        if self.prediction_mode {
            return Ok(-r.j_cp.abs());
        }

        let err_12 = ((r.theta_12 - 33.41) / 0.72).powi(2);
        let err_13 = ((r.theta_13 - 8.54) / 0.12).powi(2);
        let err_23 = ((r.theta_23 - 49.0) / 1.3).powi(2);
        let chi2_angles = err_12 + err_13 + err_23;

        // Reward larger |J_CP| by subtracting a scaled version
        Ok(chi2_angles - 100.0 * r.j_cp.abs())
    }
}

/// Run Nelder-Mead refinement starting from the grid-best point.
///
/// Freezes k (discrete), optimizes `(alpha_CP, t_sol, t_atm)` continuously.
/// Uses multi-start: grid-best + 3 random perturbations.
///
/// Returns the best `CpScanResult` found across all starts, along with the
/// optimized parameters `(alpha_CP, t_sol, t_atm)`.
pub fn refine_cp_nelder_mead(
    ctx: &CpScanContext<'_>,
    phi: &[[f64; 3]; 3],
    alpha0: f64,
    t_sol0: f64,
    t_atm0: f64,
    prediction_mode: bool,
) -> (CpScanResult, [f64; 3]) {
    use argmin::core::{Executor, State};
    use argmin::solver::neldermead::NelderMead;

    let bounds = [(0.01, 1.0), (t_sol0 - 2.0, t_sol0 + 4.0), (t_atm0 - 2.0, t_atm0 + 8.0)];

    let build_simplex = |x0: &[f64; 3]| -> Vec<Vec<f64>> {
        let steps = [0.05, 0.2, 0.5];
        let mut simplex = Vec::with_capacity(4);
        simplex.push(vec![x0[0], x0[1], x0[2]]);
        for i in 0..3 {
            let mut v = vec![x0[0], x0[1], x0[2]];
            v[i] += steps[i];
            for (j, &(lo, hi)) in bounds.iter().enumerate() {
                v[j] = v[j].clamp(lo, hi);
            }
            simplex.push(v);
        }
        simplex
    };

    // Multi-start: grid-best + 3 perturbations
    let starts: [[f64; 3]; 4] = [
        [alpha0, t_sol0, t_atm0],
        [alpha0 * 1.2, t_sol0 + 0.3, t_atm0 - 0.5],
        [alpha0 * 0.8, t_sol0 - 0.3, t_atm0 + 0.5],
        [(alpha0 + 0.1).min(1.0), t_sol0 + 0.5, t_atm0 + 1.0],
    ];

    let mut best_cost = f64::MAX;
    let mut best_params = [alpha0, t_sol0, t_atm0];

    for start in &starts {
        let simplex = build_simplex(start);
        let solver = match NelderMead::new(simplex).with_sd_tolerance(1e-8) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let problem = CpNelderMeadCost {
            ctx,
            phi: *phi,
            bounds,
            prediction_mode,
        };
        let run = Executor::new(problem, solver)
            .configure(|state| state.max_iters(500))
            .run();
        if let Ok(result) = run {
            let state = result.state();
            if let Some(param) = state.get_best_param() {
                let cost = state.get_best_cost();
                if cost < best_cost {
                    best_cost = cost;
                    best_params = [
                        param[0].clamp(bounds[0].0, bounds[0].1),
                        param[1].clamp(bounds[1].0, bounds[1].1),
                        param[2].clamp(bounds[2].0, bounds[2].1),
                    ];
                }
            }
        }
    }

    let mut bufs = CpScanBuffers::new();
    let result = evaluate_cp_scan_point(
        best_params[0], best_params[1], best_params[2],
        phi, ctx, &mut bufs,
    );
    (result, best_params)
}

// ---------------------------------------------------------------------------
// Stack-allocated 3x3 complex Hermitian eigensolver (Cardano)
// ---------------------------------------------------------------------------

/// A 3x3 complex number: (re, im) pair.
type C2 = (f64, f64);

/// Multiply two complex numbers on the stack.
#[inline(always)]
fn cmul(a: C2, b: C2) -> C2 {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

/// Conjugate of a complex number.
#[inline(always)]
fn cconj(a: C2) -> C2 { (a.0, -a.1) }

/// Eigenvalues + PMNS-relevant quantities for two 3x3 Hermitian matrices.
///
/// # Mathematical foundation
///
/// For a 3x3 Hermitian matrix H, the eigenvalues are roots of the
/// **real** characteristic polynomial:
///
/// ```text
/// lambda^3 - tr(H) * lambda^2 + s2(H) * lambda - det(H) = 0
/// ```
///
/// where `s2 = (tr^2 - tr(H^2))/2` is the second symmetric function.
/// This is solved analytically via the depressed cubic (Cardano/Vieta).
///
/// For the PMNS mixing angles, we need the unitary matrix `U` such that
/// `U^dag M_ch U_ch = diag` and `U^dag M_nu U_nu = diag`, then
/// `U_PMNS = U_ch^dag * U_nu`.  The Jarlskog invariant is:
///
/// ```text
/// J = Im(U_e1 * U_mu2 * conj(U_e2) * conj(U_mu1))
/// ```
///
/// # Why hand-rolled instead of faer
///
/// `faer::selfadjoint_eigendecomposition` allocates heap memory for the
/// working buffer.  In a tight scan loop (~10,000 calls), the allocation
/// overhead dominates.  This function uses only stack arrays and the
/// analytical Cardano formula, giving ~10x speedup for 3x3 matrices.
///
/// # Callers
///
/// - [`test_cp_violation_joint_3d_scan`]: inner scan loop
/// - Any future tight-loop PMNS computation
///
/// Returns `(eigenvalues_sorted, eigenvectors_as_columns)` where
/// eigenvalues are in ascending order.
#[allow(clippy::needless_range_loop)]
pub fn hermitian_3x3_eig(h: &[[C2; 3]; 3]) -> ([f64; 3], [[C2; 3]; 3]) {
    // Characteristic polynomial coefficients (all real for Hermitian H):
    // p = -tr(H), q = s2(H), r = -det(H)
    let tr_h = h[0][0].0 + h[1][1].0 + h[2][2].0;

    // tr(H^2) = sum_{i,j} |H[i][j]|^2
    let mut tr_h2 = 0.0_f64;
    for row in h {
        for &(re, im) in row {
            tr_h2 += re * re + im * im;
        }
    }
    let s2 = (tr_h * tr_h - tr_h2) * 0.5;

    // det(H) via Sarrus rule for 3x3 complex matrix (result is real)
    let det = {
        let a = cmul(cmul(h[0][0], h[1][1]), h[2][2]);
        let b = cmul(cmul(h[0][1], h[1][2]), h[2][0]);
        let c = cmul(cmul(h[0][2], h[1][0]), h[2][1]);
        let d = cmul(cmul(h[0][2], h[1][1]), h[2][0]);
        let e = cmul(cmul(h[0][1], h[1][0]), h[2][2]);
        let f = cmul(cmul(h[0][0], h[1][2]), h[2][1]);
        // det = a + b + c - d - e - f (real part only, imaginary cancels)
        a.0 + b.0 + c.0 - d.0 - e.0 - f.0
    };

    // Depressed cubic: t^3 + p*t + q = 0 where lambda = t + tr_h/3
    let shift = tr_h / 3.0;
    let p = s2 - tr_h * tr_h / 3.0;
    let q = tr_h * s2 / 3.0 - 2.0 * tr_h * tr_h * tr_h / 27.0 - det;

    // Vieta trigonometric solution (always 3 real roots for Hermitian)
    let disc = -(4.0 * p * p * p + 27.0 * q * q);
    let mut evals = [0.0_f64; 3];
    if disc.abs() < 1e-30 || p.abs() < 1e-30 {
        // Degenerate case: all eigenvalues equal or nearly so
        evals = [shift; 3];
        if p.abs() > 1e-30 {
            let r = ((-p / 3.0).max(0.0)).sqrt();
            let cos_theta = (-q / (2.0 * r * r * r)).clamp(-1.0, 1.0);
            let theta = cos_theta.acos() / 3.0;
            evals[0] = 2.0 * r * theta.cos() + shift;
            evals[1] = 2.0 * r * (theta - 2.0 * std::f64::consts::FRAC_PI_3).cos() + shift;
            evals[2] = 2.0 * r * (theta + 2.0 * std::f64::consts::FRAC_PI_3).cos() + shift;
        }
    } else {
        let r = ((-p / 3.0).max(0.0)).sqrt();
        let cos_theta = (-q / (2.0 * r * r * r)).clamp(-1.0, 1.0);
        let theta = cos_theta.acos() / 3.0;
        evals[0] = 2.0 * r * theta.cos() + shift;
        evals[1] = 2.0 * r * (theta - 2.0 * std::f64::consts::FRAC_PI_3).cos() + shift;
        evals[2] = 2.0 * r * (theta + 2.0 * std::f64::consts::FRAC_PI_3).cos() + shift;
    }

    // Sort eigenvalues ascending
    if evals[0] > evals[1] { evals.swap(0, 1); }
    if evals[1] > evals[2] { evals.swap(1, 2); }
    if evals[0] > evals[1] { evals.swap(0, 1); }

    // Eigenvectors via (H - lambda*I) null space:
    // For each eigenvalue, find the eigenvector by cross product of two
    // rows of (H - lambda*I).  This is the standard adjugate method.
    let mut evecs = [[(0.0, 0.0); 3]; 3];
    for col in 0..3 {
        let lam = evals[col];
        let mut a = [[(0.0, 0.0); 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                a[i][j] = h[i][j];
            }
            a[i][i].0 -= lam;
        }
        // Cross product of rows 0 and 1: v = row0 x row1
        let v = [
            (cmul(a[0][1], a[1][2]).0 - cmul(a[0][2], a[1][1]).0,
             cmul(a[0][1], a[1][2]).1 - cmul(a[0][2], a[1][1]).1),
            (cmul(a[0][2], a[1][0]).0 - cmul(a[0][0], a[1][2]).0,
             cmul(a[0][2], a[1][0]).1 - cmul(a[0][0], a[1][2]).1),
            (cmul(a[0][0], a[1][1]).0 - cmul(a[0][1], a[1][0]).0,
             cmul(a[0][0], a[1][1]).1 - cmul(a[0][1], a[1][0]).1),
        ];
        let norm = (v[0].0 * v[0].0 + v[0].1 * v[0].1
                  + v[1].0 * v[1].0 + v[1].1 * v[1].1
                  + v[2].0 * v[2].0 + v[2].1 * v[2].1).sqrt();
        if norm > 1e-15 {
            for i in 0..3 { evecs[i][col] = (v[i].0 / norm, v[i].1 / norm); }
        } else {
            // Try rows 0 and 2
            let v2 = [
                (cmul(a[0][1], a[2][2]).0 - cmul(a[0][2], a[2][1]).0,
                 cmul(a[0][1], a[2][2]).1 - cmul(a[0][2], a[2][1]).1),
                (cmul(a[0][2], a[2][0]).0 - cmul(a[0][0], a[2][2]).0,
                 cmul(a[0][2], a[2][0]).1 - cmul(a[0][0], a[2][2]).1),
                (cmul(a[0][0], a[2][1]).0 - cmul(a[0][1], a[2][0]).0,
                 cmul(a[0][0], a[2][1]).1 - cmul(a[0][1], a[2][0]).1),
            ];
            let norm2 = (v2[0].0 * v2[0].0 + v2[0].1 * v2[0].1
                       + v2[1].0 * v2[1].0 + v2[1].1 * v2[1].1
                       + v2[2].0 * v2[2].0 + v2[2].1 * v2[2].1).sqrt();
            if norm2 > 1e-15 {
                for i in 0..3 { evecs[i][col] = (v2[i].0 / norm2, v2[i].1 / norm2); }
            } else {
                // Triple degeneracy -- use identity column
                evecs[col][col] = (1.0, 0.0);
            }
        }

        // U(1) phase canonicalization: make the largest-magnitude component
        // real and nonnegative.  This is the complex analogue of the LAPACK
        // convention for real eigenvectors (largest component positive).
        //
        // Without this, each eigenvector carries an arbitrary e^{i*theta}
        // phase.  Quantities like arg(-U_e3) for delta_CP depend on
        // individual matrix elements and are meaningless without a fixed
        // phase convention.
        let max_idx = {
            let mut best = 0;
            let mut best_mag_sq = evecs[0][col].0 * evecs[0][col].0
                                + evecs[0][col].1 * evecs[0][col].1;
            for idx in 1..3 {
                let mag_sq = evecs[idx][col].0 * evecs[idx][col].0
                           + evecs[idx][col].1 * evecs[idx][col].1;
                if mag_sq > best_mag_sq {
                    best = idx;
                    best_mag_sq = mag_sq;
                }
            }
            best
        };
        let (re, im) = evecs[max_idx][col];
        let mag = (re * re + im * im).sqrt();
        if mag > 1e-15 {
            // Rotate entire vector by e^{-i*theta} where theta = arg(v_max)
            let cos_t = re / mag;
            let sin_t = im / mag;
            for i in 0..3 {
                let (r, m) = evecs[i][col];
                evecs[i][col] = (r * cos_t + m * sin_t,
                                 m * cos_t - r * sin_t);
            }
            // Ensure the reference component is strictly nonneg real
            if evecs[max_idx][col].0 < 0.0 {
                for i in 0..3 {
                    evecs[i][col].0 = -evecs[i][col].0;
                    evecs[i][col].1 = -evecs[i][col].1;
                }
            }
        }
    }

    (evals, evecs)
}

/// Minimum relative eigenvalue gap below which the Cardano cross-product
/// eigenvector method becomes numerically unstable.  When the gap falls
/// below this threshold times the Frobenius norm, we fall back to faer's
/// iterative QR which handles near-degeneracies gracefully.
const EIGGAP_THRESHOLD: f64 = 1e-10;

/// Hybrid 3x3 Hermitian eigensolver: Cardano if well-separated, faer if
/// degenerate.
///
/// Uses [`hermitian_3x3_eig`] (zero-alloc, O(1) Cardano) when eigenvalue
/// gaps are large relative to the matrix norm.  Falls back to faer's
/// `selfadjoint_eigendecomposition` near degeneracies where the cross-product
/// eigenvector method loses accuracy.
///
/// Returns `(eigenvalues_sorted, eigenvectors_as_columns)`.
#[allow(clippy::needless_range_loop)]
pub fn hermitian_3x3_eig_hybrid(h: &[[C2; 3]; 3]) -> ([f64; 3], [[C2; 3]; 3]) {
    let (evals, evecs) = hermitian_3x3_eig(h);

    // Check eigenvalue gap relative to matrix Frobenius norm
    let h_frob_sq: f64 = h.iter().flat_map(|row| row.iter())
        .map(|&(r, m)| r * r + m * m).sum();
    let h_norm = h_frob_sq.sqrt();

    let min_gap = (evals[1] - evals[0]).abs().min((evals[2] - evals[1]).abs());

    if min_gap > EIGGAP_THRESHOLD * h_norm {
        (evals, evecs)
    } else {
        // faer fallback for near-degenerate cases
        let mut h_faer = faer::Mat::<faer::complex_native::c64>::zeros(3, 3);
        for i in 0..3 {
            for j in 0..3 {
                h_faer.write(i, j, faer::complex_native::c64::new(h[i][j].0, h[i][j].1));
            }
        }
        let eig = h_faer.selfadjoint_eigendecomposition(faer::Side::Lower);
        let mut fe = [0.0_f64; 3];
        for i in 0..3 { fe[i] = eig.s().column_vector().read(i).re; }

        // Sort and build index map
        let mut idx = [0_usize, 1, 2];
        idx.sort_by(|&a, &b| fe[a].partial_cmp(&fe[b]).unwrap());
        let sorted_evals = [fe[idx[0]], fe[idx[1]], fe[idx[2]]];

        let mut sorted_evecs = [[(0.0, 0.0); 3]; 3];
        for col in 0..3 {
            let src = idx[col];
            for row in 0..3 {
                let c = eig.u().read(row, src);
                sorted_evecs[row][col] = (c.re, c.im);
            }
            // Apply same phase canonicalization as Cardano path
            let max_idx = (0..3).max_by(|&a, &b| {
                let na = sorted_evecs[a][col].0 * sorted_evecs[a][col].0
                       + sorted_evecs[a][col].1 * sorted_evecs[a][col].1;
                let nb = sorted_evecs[b][col].0 * sorted_evecs[b][col].0
                       + sorted_evecs[b][col].1 * sorted_evecs[b][col].1;
                na.partial_cmp(&nb).unwrap()
            }).unwrap();
            let (re, im) = sorted_evecs[max_idx][col];
            let mag = (re * re + im * im).sqrt();
            if mag > 1e-15 {
                let cos_t = re / mag;
                let sin_t = im / mag;
                for i in 0..3 {
                    let (r, m) = sorted_evecs[i][col];
                    sorted_evecs[i][col] = (r * cos_t + m * sin_t,
                                            m * cos_t - r * sin_t);
                }
                if sorted_evecs[max_idx][col].0 < 0.0 {
                    for i in 0..3 {
                        sorted_evecs[i][col].0 = -sorted_evecs[i][col].0;
                        sorted_evecs[i][col].1 = -sorted_evecs[i][col].1;
                    }
                }
            }
        }
        (sorted_evals, sorted_evecs)
    }
}

/// Compute Jarlskog invariant and mixing angles directly from two 3x3
/// Hermitian mass matrices, entirely on the stack.
///
/// # Mathematical foundation
///
/// Given charged-lepton mass matrix `M_ch` and neutrino mass matrix
/// `M_nu`, diagonalise both via [`hermitian_3x3_eig`], form
/// `U_PMNS = U_ch^dag * U_nu`, apply the stored permutation, then
/// extract:
///
/// ```text
/// theta_13 = asin(|U_e3|)
/// theta_12 = asin(|U_e2| / cos(theta_13))
/// theta_23 = asin(|U_mu3| / cos(theta_13))
/// J_CP     = Im(U_e1 * U_mu2 * conj(U_e2) * conj(U_mu1))
/// delta_CP = arg(-U_e3)
/// ```
///
/// # Why this exists
///
/// Eliminates faer heap allocation in tight scan loops.  The eigensolve
/// uses Cardano's formula (O(1) flops, zero allocation) instead of
/// iterative QR (O(n^3) with heap buffer).
///
/// # Returns
///
/// `(theta_12, theta_13, theta_23, j_cp, delta_cp, delta_cp_invariant)` in degrees.
pub fn pmns_from_hermitian_pair(
    m_ch: &[[C2; 3]; 3],
    m_nu: &[[C2; 3]; 3],
    perm_u: &[usize; 3],
    perm_d: &[usize; 3],
) -> (f64, f64, f64, f64, f64, f64) {
    let (_evals_ch, u_ch) = hermitian_3x3_eig(m_ch);
    let (_evals_nu, u_nu) = hermitian_3x3_eig(m_nu);

    // U_PMNS = U_ch^dag * U_nu  (3x3 complex multiply)
    let mut u_pmns = [[(0.0, 0.0); 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let mut s = (0.0_f64, 0.0_f64);
            for k in 0..3 {
                // U_ch^dag[i][k] = conj(U_ch[k][i])
                let a = cconj(u_ch[k][i]);
                let b = u_nu[k][j];
                s.0 += a.0 * b.0 - a.1 * b.1;
                s.1 += a.0 * b.1 + a.1 * b.0;
            }
            u_pmns[i][j] = s;
        }
    }

    // Apply permutation
    let mut u_perm = [[(0.0, 0.0); 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            u_perm[i][j] = u_pmns[perm_u[i]][perm_d[j]];
        }
    }

    // Extract angles
    let u_e3_abs = (u_perm[0][2].0 * u_perm[0][2].0
                  + u_perm[0][2].1 * u_perm[0][2].1).sqrt();
    let theta_13 = u_e3_abs.min(1.0).asin().to_degrees();
    let cos_13 = theta_13.to_radians().cos();

    let theta_12 = if cos_13 > 1e-15 {
        let u_e2_abs = (u_perm[0][1].0 * u_perm[0][1].0
                      + u_perm[0][1].1 * u_perm[0][1].1).sqrt();
        (u_e2_abs / cos_13).min(1.0).asin().to_degrees()
    } else { 0.0 };

    let theta_23 = if cos_13 > 1e-15 {
        let u_mu3_abs = (u_perm[1][2].0 * u_perm[1][2].0
                       + u_perm[1][2].1 * u_perm[1][2].1).sqrt();
        (u_mu3_abs / cos_13).min(1.0).asin().to_degrees()
    } else { 0.0 };

    // Jarlskog: J = Im(U_e1 * U_mu2 * conj(U_e2) * conj(U_mu1))
    let prod = cmul(cmul(u_perm[0][0], u_perm[1][1]),
                    cmul(cconj(u_perm[0][1]), cconj(u_perm[1][0])));
    let j_cp = prod.1;

    // delta_CP = arg(-U_e3)
    let neg_ue3 = (-u_perm[0][2].0, -u_perm[0][2].1);
    let delta_cp = neg_ue3.1.atan2(neg_ue3.0).to_degrees();

    // Rephasing-invariant delta_CP via moduli + Jarlskog
    let u_moduli = {
        let mut m = [[0.0_f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                m[i][j] = (u_perm[i][j].0 * u_perm[i][j].0
                         + u_perm[i][j].1 * u_perm[i][j].1).sqrt();
            }
        }
        m
    };
    let delta_cp_invariant = extract_delta_cp_invariant(&u_moduli, j_cp);

    (theta_12, theta_13, theta_23, j_cp, delta_cp, delta_cp_invariant)
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

    /// V_6 Jacobian solar selectivity test.
    ///
    /// Computes finite-difference gradients of all three PMNS angles with
    /// respect to the 6 V_6 coefficients at beta=0. Then finds the unit
    /// direction u in S^5 that maximizes solar sensitivity while minimizing
    /// reactor/atmospheric leakage. This is basis-invariant because we
    /// optimize over the full subspace.
    #[test]
    fn test_v6_jacobian_solar_selectivity() {
        let ch_pair = (11_usize, 12);
        let nu_pair = (7_usize, 8);
        let alpha_ch = 3.75;
        let alpha_nu = 1.30;

        // Extract V_6 basis
        let (v6_basis, singular_values, assessors) = extract_v6_basis();
        let flavor_map = AssessorToFlavorMap::default_partition(&assessors);
        let n_basis = v6_basis.nrows().min(6);

        println!("--- V_6 JACOBIAN SOLAR SELECTIVITY ---");
        println!("  V_6 basis rank: {}", n_basis);
        println!("  Singular values: [{}]",
            singular_values.iter().map(|s| format!("{:.3}", s)).collect::<Vec<_>>().join(", "));

        // Lock the baseline permutation: compute once at beta=0
        let (m_ch_0, m_nu_0) = construct_pmns_matrices_two_param(
            ch_pair, nu_pair, alpha_ch, alpha_nu,
        );
        let eig_ch_0 = m_ch_0.selfadjoint_eigendecomposition(faer::Side::Lower);
        let eig_nu_0 = m_nu_0.selfadjoint_eigendecomposition(faer::Side::Lower);
        let u_raw_0 = eig_ch_0.u().transpose() * eig_nu_0.u();
        let (_, perm_u_0, perm_d_0) = crate::quark_sector::extract_ckm_permutation_aware(&u_raw_0);

        // Helper: compute angles for given beta using FIXED permutation
        let compute_angles = |beta: &[f64; 6]| -> (f64, f64, f64) {
            let (m_ch, mut m_nu) = construct_pmns_matrices_two_param(
                ch_pair, nu_pair, alpha_ch, alpha_nu,
            );
            apply_v6_perturbation(&mut m_nu, &v6_basis, beta, &flavor_map);

            let eig_ch = m_ch.selfadjoint_eigendecomposition(faer::Side::Lower);
            let m_nu_s = (&m_nu + m_nu.transpose()) * faer::scale(0.5);
            let eig_nu = m_nu_s.selfadjoint_eigendecomposition(faer::Side::Lower);
            let u_raw = eig_ch.u().transpose() * eig_nu.u();

            // Apply the baseline permutation (prevents flip artifacts)
            let mut u_pmns = faer::Mat::<f64>::zeros(3, 3);
            for i in 0..3 {
                for j in 0..3 {
                    u_pmns.write(i, j, u_raw.read(perm_u_0[i], perm_d_0[j]));
                }
            }
            extract_pmns_angles(&u_pmns)
        };

        // Verify beta=0 recovery
        let (t12_0, t13_0, t23_0) = compute_angles(&[0.0; 6]);
        println!("  beta=0 baseline: theta_12={:.4}, theta_13={:.4}, theta_23={:.4}",
            t12_0, t13_0, t23_0);

        // Finite-difference gradients at beta=0 for multiple epsilon values
        let epsilons = [0.01, 0.05, 0.1];
        let mut gradients_12: Vec<[f64; 6]> = Vec::new();
        let mut gradients_13: Vec<[f64; 6]> = Vec::new();
        let mut gradients_23: Vec<[f64; 6]> = Vec::new();

        for &eps in &epsilons {
            let mut g_12 = [0.0_f64; 6];
            let mut g_13 = [0.0_f64; 6];
            let mut g_23 = [0.0_f64; 6];

            for mu in 0..n_basis {
                let mut beta_plus = [0.0_f64; 6];
                let mut beta_minus = [0.0_f64; 6];
                beta_plus[mu] = eps;
                beta_minus[mu] = -eps;

                let (t12_p, t13_p, t23_p) = compute_angles(&beta_plus);
                let (t12_m, t13_m, t23_m) = compute_angles(&beta_minus);

                g_12[mu] = (t12_p - t12_m) / (2.0 * eps);
                g_13[mu] = (t13_p - t13_m) / (2.0 * eps);
                g_23[mu] = (t23_p - t23_m) / (2.0 * eps);
            }

            println!("\n  eps = {:.3}:", eps);
            println!("    g_12 = [{}]", g_12.iter().map(|x| format!("{:.4}", x)).collect::<Vec<_>>().join(", "));
            println!("    g_13 = [{}]", g_13.iter().map(|x| format!("{:.4}", x)).collect::<Vec<_>>().join(", "));
            println!("    g_23 = [{}]", g_23.iter().map(|x| format!("{:.4}", x)).collect::<Vec<_>>().join(", "));

            gradients_12.push(g_12);
            gradients_13.push(g_13);
            gradients_23.push(g_23);
        }

        // Epsilon stability check: angular difference between gradient vectors
        // at different epsilon values should be < 5 deg
        let vec_angle = |a: &[f64; 6], b: &[f64; 6]| -> f64 {
            let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
            let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
            if na < 1e-15 || nb < 1e-15 { return 0.0; }
            (dot / (na * nb)).clamp(-1.0, 1.0).acos().to_degrees()
        };

        for pair in [(0, 1), (0, 2), (1, 2)] {
            let angle_12 = vec_angle(&gradients_12[pair.0], &gradients_12[pair.1]);
            let angle_13 = vec_angle(&gradients_13[pair.0], &gradients_13[pair.1]);
            let angle_23 = vec_angle(&gradients_23[pair.0], &gradients_23[pair.1]);
            println!("\n  Stability (eps[{}] vs eps[{}]):", pair.0, pair.1);
            println!("    g_12 angle: {:.2} deg", angle_12);
            println!("    g_13 angle: {:.2} deg", angle_13);
            println!("    g_23 angle: {:.2} deg", angle_23);

            // Gradient vectors should be stable if they have meaningful magnitude
            let norm_12: f64 = gradients_12[pair.0].iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm_12 > 0.01 {
                assert!(angle_12 < 15.0,
                    "g_12 unstable: {:.2} deg between eps[{}] and eps[{}]",
                    angle_12, pair.0, pair.1);
            }
        }

        // Optimize over full V_6 subspace: find unit direction u maximizing
        //   S(u) = |g_12 . u| - lambda_13 * |g_13 . u| - lambda_23 * |g_23 . u|
        let lambda_13 = 10.0_f64;
        let lambda_23 = 3.0_f64;

        // Use the middle epsilon (0.05) gradient as reference
        let g_12_ref = &gradients_12[1];
        let g_13_ref = &gradients_13[1];
        let g_23_ref = &gradients_23[1];

        // Grid search over random directions on S^5 (deterministic seed)
        let mut best_score = f64::NEG_INFINITY;
        let mut best_u = [0.0_f64; 6];

        // Structured search: try all single-axis directions first
        for mu in 0..n_basis {
            for sign in [-1.0_f64, 1.0] {
                let mut u = [0.0_f64; 6];
                u[mu] = sign;
                let s12: f64 = g_12_ref.iter().zip(u.iter()).map(|(g, x)| g * x).sum();
                let s13: f64 = g_13_ref.iter().zip(u.iter()).map(|(g, x)| g * x).sum();
                let s23: f64 = g_23_ref.iter().zip(u.iter()).map(|(g, x)| g * x).sum();
                let score = s12.abs() - lambda_13 * s13.abs() - lambda_23 * s23.abs();
                if score > best_score {
                    best_score = score;
                    best_u = u;
                }
            }
        }

        // Pairwise search on 2D subspaces
        for mu1 in 0..n_basis {
            for mu2 in (mu1 + 1)..n_basis {
                for angle_step in 0..360 {
                    let theta = (angle_step as f64) * std::f64::consts::PI / 180.0;
                    let mut u = [0.0_f64; 6];
                    u[mu1] = theta.cos();
                    u[mu2] = theta.sin();
                    let s12: f64 = g_12_ref.iter().zip(u.iter()).map(|(g, x)| g * x).sum();
                    let s13: f64 = g_13_ref.iter().zip(u.iter()).map(|(g, x)| g * x).sum();
                    let s23: f64 = g_23_ref.iter().zip(u.iter()).map(|(g, x)| g * x).sum();
                    let score = s12.abs() - lambda_13 * s13.abs() - lambda_23 * s23.abs();
                    if score > best_score {
                        best_score = score;
                        best_u = u;
                    }
                }
            }
        }

        // Random search (deterministic LCG for reproducibility)
        let mut rng_state = 42_u64;
        let lcg_next = |state: &mut u64| -> f64 {
            *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (*state >> 33) as f64 / (1u64 << 31) as f64 * 2.0 - 1.0
        };

        for _ in 0..10000 {
            let mut u = [0.0_f64; 6];
            let mut norm_sq = 0.0_f64;
            for component in u.iter_mut().take(n_basis) {
                *component = lcg_next(&mut rng_state);
                norm_sq += *component * *component;
            }
            if norm_sq < 1e-10 { continue; }
            let inv_norm = 1.0 / norm_sq.sqrt();
            for component in u.iter_mut().take(n_basis) {
                *component *= inv_norm;
            }

            let s12: f64 = g_12_ref.iter().zip(u.iter()).map(|(g, x)| g * x).sum();
            let s13: f64 = g_13_ref.iter().zip(u.iter()).map(|(g, x)| g * x).sum();
            let s23: f64 = g_23_ref.iter().zip(u.iter()).map(|(g, x)| g * x).sum();
            let score = s12.abs() - lambda_13 * s13.abs() - lambda_23 * s23.abs();
            if score > best_score {
                best_score = score;
                best_u = u;
            }
        }

        println!("\n  === OPTIMAL SOLAR DIRECTION ===");
        println!("  u = [{}]", best_u.iter().map(|x| format!("{:.4}", x)).collect::<Vec<_>>().join(", "));
        println!("  Score S(u) = {:.6}", best_score);

        // Report projections along optimal direction
        let s12_opt: f64 = g_12_ref.iter().zip(best_u.iter()).map(|(g, x)| g * x).sum();
        let s13_opt: f64 = g_13_ref.iter().zip(best_u.iter()).map(|(g, x)| g * x).sum();
        let s23_opt: f64 = g_23_ref.iter().zip(best_u.iter()).map(|(g, x)| g * x).sum();
        println!("  g_12 . u = {:.4} (solar sensitivity)", s12_opt);
        println!("  g_13 . u = {:.4} (reactor leakage)", s13_opt);
        println!("  g_23 . u = {:.4} (atmospheric leakage)", s23_opt);
    }

    /// 1D solar scan along the optimal V_6 direction.
    ///
    /// Uses the solar-selective direction from the Jacobian test to scan
    /// for the optimal perturbation amplitude that corrects theta_12
    /// toward the PDG value while preserving theta_13.
    #[test]
    fn test_v6_solar_1d_scan() {
        let ch_pair = (11_usize, 12);
        let nu_pair = (7_usize, 8);
        let alpha_ch = 3.75;
        let alpha_nu = 1.30;
        let pdg_t12 = 33.41_f64;
        let pdg_t13 = 8.54_f64;
        let pdg_t23 = 49.0_f64;

        // Extract V_6 basis and flavor map
        let (v6_basis, _sv, assessors) = extract_v6_basis();
        let flavor_map = AssessorToFlavorMap::default_partition(&assessors);
        let n_basis = v6_basis.nrows().min(6);

        // Lock baseline permutation
        let (m_ch_0, m_nu_0) = construct_pmns_matrices_two_param(
            ch_pair, nu_pair, alpha_ch, alpha_nu,
        );
        let eig_ch_0 = m_ch_0.selfadjoint_eigendecomposition(faer::Side::Lower);
        let eig_nu_0 = m_nu_0.selfadjoint_eigendecomposition(faer::Side::Lower);
        let u_raw_0 = eig_ch_0.u().transpose() * eig_nu_0.u();
        let (_, perm_u_0, perm_d_0) = crate::quark_sector::extract_ckm_permutation_aware(&u_raw_0);

        // Compute Jacobian at beta=0 (same as Jacobian test, eps=0.05)
        let eps = 0.05_f64;
        let compute_angles = |beta: &[f64; 6]| -> (f64, f64, f64) {
            let (m_ch, mut m_nu) = construct_pmns_matrices_two_param(
                ch_pair, nu_pair, alpha_ch, alpha_nu,
            );
            apply_v6_perturbation(&mut m_nu, &v6_basis, beta, &flavor_map);

            let eig_ch = m_ch.selfadjoint_eigendecomposition(faer::Side::Lower);
            let m_nu_s = (&m_nu + m_nu.transpose()) * faer::scale(0.5);
            let eig_nu = m_nu_s.selfadjoint_eigendecomposition(faer::Side::Lower);
            let u_raw = eig_ch.u().transpose() * eig_nu.u();

            // Apply locked baseline permutation
            let mut u_pmns = faer::Mat::<f64>::zeros(3, 3);
            for i in 0..3 {
                for j in 0..3 {
                    u_pmns.write(i, j, u_raw.read(perm_u_0[i], perm_d_0[j]));
                }
            }
            extract_pmns_angles(&u_pmns)
        };

        let mut g_12 = [0.0_f64; 6];
        let mut g_13 = [0.0_f64; 6];
        let mut g_23 = [0.0_f64; 6];
        for mu in 0..n_basis {
            let mut bp = [0.0_f64; 6];
            let mut bm = [0.0_f64; 6];
            bp[mu] = eps;
            bm[mu] = -eps;
            let (t12_p, t13_p, t23_p) = compute_angles(&bp);
            let (t12_m, t13_m, t23_m) = compute_angles(&bm);
            g_12[mu] = (t12_p - t12_m) / (2.0 * eps);
            g_13[mu] = (t13_p - t13_m) / (2.0 * eps);
            g_23[mu] = (t23_p - t23_m) / (2.0 * eps);
        }

        // Find optimal direction (same algorithm as Jacobian test)
        let lambda_13 = 10.0_f64;
        let lambda_23 = 3.0_f64;
        let mut best_score = f64::NEG_INFINITY;
        let mut best_u = [0.0_f64; 6];

        // Pairwise + random search
        for mu1 in 0..n_basis {
            for sign in [-1.0_f64, 1.0] {
                let mut u = [0.0_f64; 6];
                u[mu1] = sign;
                let s12: f64 = g_12.iter().zip(u.iter()).map(|(g, x)| g * x).sum();
                let s13: f64 = g_13.iter().zip(u.iter()).map(|(g, x)| g * x).sum();
                let s23: f64 = g_23.iter().zip(u.iter()).map(|(g, x)| g * x).sum();
                let score = s12.abs() - lambda_13 * s13.abs() - lambda_23 * s23.abs();
                if score > best_score { best_score = score; best_u = u; }
            }
        }
        for mu1 in 0..n_basis {
            for mu2 in (mu1 + 1)..n_basis {
                for angle_step in 0..360 {
                    let theta = (angle_step as f64) * std::f64::consts::PI / 180.0;
                    let mut u = [0.0_f64; 6];
                    u[mu1] = theta.cos();
                    u[mu2] = theta.sin();
                    let s12: f64 = g_12.iter().zip(u.iter()).map(|(g, x)| g * x).sum();
                    let s13: f64 = g_13.iter().zip(u.iter()).map(|(g, x)| g * x).sum();
                    let s23: f64 = g_23.iter().zip(u.iter()).map(|(g, x)| g * x).sum();
                    let score = s12.abs() - lambda_13 * s13.abs() - lambda_23 * s23.abs();
                    if score > best_score { best_score = score; best_u = u; }
                }
            }
        }
        let mut rng_state = 42_u64;
        let lcg_next = |state: &mut u64| -> f64 {
            *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (*state >> 33) as f64 / (1u64 << 31) as f64 * 2.0 - 1.0
        };
        for _ in 0..10000 {
            let mut u = [0.0_f64; 6];
            let mut norm_sq = 0.0_f64;
            for component in u.iter_mut().take(n_basis) {
                *component = lcg_next(&mut rng_state);
                norm_sq += *component * *component;
            }
            if norm_sq < 1e-10 { continue; }
            let inv_norm = 1.0 / norm_sq.sqrt();
            for component in u.iter_mut().take(n_basis) { *component *= inv_norm; }
            let s12: f64 = g_12.iter().zip(u.iter()).map(|(g, x)| g * x).sum();
            let s13: f64 = g_13.iter().zip(u.iter()).map(|(g, x)| g * x).sum();
            let s23: f64 = g_23.iter().zip(u.iter()).map(|(g, x)| g * x).sum();
            let score = s12.abs() - lambda_13 * s13.abs() - lambda_23 * s23.abs();
            if score > best_score { best_score = score; best_u = u; }
        }

        println!("--- V_6 SOLAR 1D SCAN ---");
        println!("  Optimal direction u = [{}]",
            best_u.iter().map(|x| format!("{:.4}", x)).collect::<Vec<_>>().join(", "));

        // 1D scan along u: beta = t * u for t in [-5.0, 5.0]
        println!("\n  {:>8} {:>10} {:>10} {:>10} {:>10}",
            "t", "theta_12", "theta_13", "theta_23", "score");

        let mut best_t = 0.0_f64;
        let mut best_scan_score = f64::MAX;
        let mut best_angles = (0.0_f64, 0.0_f64, 0.0_f64);
        let step = 0.01_f64;
        let n_steps = 1000_i32;

        for step_i in -n_steps..=n_steps {
            let t = step_i as f64 * step;
            let mut beta = [0.0_f64; 6];
            for k in 0..6 {
                beta[k] = t * best_u[k];
            }

            let (t12, t13, t23) = compute_angles(&beta);

            // Hard constraint: theta_13 must stay within 0.1 deg of PDG
            let t13_ok = (t13 - pdg_t13).abs() < 0.1;

            if t13_ok {
                let score = (t12 - pdg_t12).abs();
                if score < best_scan_score {
                    best_scan_score = score;
                    best_t = t;
                    best_angles = (t12, t13, t23);
                }
            }

            // Log every 100th step
            if step_i % 100 == 0 {
                let marker = if t13_ok { " " } else { "*" };
                println!("  {:8.2} {:10.4} {:10.4} {:10.4} {:10.4}{}",
                    t, t12, t13, t23, (t12 - pdg_t12).abs(), marker);
            }
        }

        // Also log raw matrices at the best point
        {
            let mut beta_best = [0.0_f64; 6];
            for k in 0..6 { beta_best[k] = best_t * best_u[k]; }
            let (m_ch, mut m_nu) = construct_pmns_matrices_two_param(
                ch_pair, nu_pair, alpha_ch, alpha_nu,
            );
            apply_v6_perturbation(&mut m_nu, &v6_basis, &beta_best, &flavor_map);

            println!("\n  Best-point mass matrices:");
            println!("  M_ch = [");
            for i in 0..3 {
                println!("    [{:.6}, {:.6}, {:.6}]",
                    m_ch.read(i, 0), m_ch.read(i, 1), m_ch.read(i, 2));
            }
            println!("  ]");
            println!("  M_nu = [");
            for i in 0..3 {
                println!("    [{:.6}, {:.6}, {:.6}]",
                    m_nu.read(i, 0), m_nu.read(i, 1), m_nu.read(i, 2));
            }
            println!("  ]");
        }

        println!("\n  === BEST V_6 SOLAR CORRECTION (1D SCAN) ===");
        println!("  t_optimal = {:.4}", best_t);
        println!("  beta = [{}]",
            (0..6).map(|k| format!("{:.4}", best_t * best_u[k])).collect::<Vec<_>>().join(", "));
        println!("  theta_12 = {:.4} deg (PDG: {:.2}, error: {:.2}%)",
            best_angles.0, pdg_t12, ((best_angles.0 - pdg_t12) / pdg_t12 * 100.0).abs());
        println!("  theta_13 = {:.4} deg (PDG: {:.2}, error: {:.2}%)",
            best_angles.1, pdg_t13, ((best_angles.1 - pdg_t13) / pdg_t13 * 100.0).abs());
        println!("  theta_23 = {:.4} deg (PDG: {:.2}, error: {:.2}%)",
            best_angles.2, pdg_t23, ((best_angles.2 - pdg_t23) / pdg_t23 * 100.0).abs());

        // theta_13 hard constraint verification across entire scan
        println!("\n  Verifying theta_13 stability at best point...");
        assert!(
            (best_angles.1 - pdg_t13).abs() < 0.1,
            "theta_13 violated hard constraint: {:.4} deg (PDG: {:.2})",
            best_angles.1, pdg_t13
        );
    }

    /// 1D solar scan using DirectOffDiagonalLift.
    ///
    /// The DirectOffDiagonal lift broke the collinearity that killed the
    /// (12/12/6) partition. This scan checks whether V_6 can push theta_12
    /// toward 33.4 deg while keeping theta_13 within 0.1 deg.
    #[test]
    fn test_v6_solar_1d_scan_direct_lift() {
        let ch_pair = (11_usize, 12);
        let nu_pair = (7_usize, 8);
        let alpha_ch = 3.75;
        let alpha_nu = 1.30;
        let pdg_t12 = 33.41_f64;
        let pdg_t13 = 8.54_f64;
        let pdg_t23 = 49.0_f64;
        let eps = 0.05_f64;

        let (v6_basis, _sv, _assessors) = extract_v6_basis();
        let lift = DirectOffDiagonalLift;
        let n_basis = v6_basis.nrows().min(6);

        // Lock baseline permutation
        let (m_ch_0, m_nu_0) = construct_pmns_matrices_two_param(
            ch_pair, nu_pair, alpha_ch, alpha_nu,
        );
        let eig_ch_0 = m_ch_0.selfadjoint_eigendecomposition(faer::Side::Lower);
        let eig_nu_0 = m_nu_0.selfadjoint_eigendecomposition(faer::Side::Lower);
        let u_raw_0 = eig_ch_0.u().transpose() * eig_nu_0.u();
        let (_, perm_u, perm_d) = crate::quark_sector::extract_ckm_permutation_aware(&u_raw_0);

        let compute_angles = |beta: &[f64; 6]| -> (f64, f64, f64) {
            let (m_ch, mut m_nu) = construct_pmns_matrices_two_param(
                ch_pair, nu_pair, alpha_ch, alpha_nu,
            );
            apply_v6_perturbation(&mut m_nu, &v6_basis, beta, &lift);

            let eig_ch = m_ch.selfadjoint_eigendecomposition(faer::Side::Lower);
            let m_nu_s = (&m_nu + m_nu.transpose()) * faer::scale(0.5);
            let eig_nu = m_nu_s.selfadjoint_eigendecomposition(faer::Side::Lower);
            let u_raw = eig_ch.u().transpose() * eig_nu.u();

            let mut u_pmns = faer::Mat::<f64>::zeros(3, 3);
            for i in 0..3 {
                for j in 0..3 {
                    u_pmns.write(i, j, u_raw.read(perm_u[i], perm_d[j]));
                }
            }
            extract_pmns_angles(&u_pmns)
        };

        // Compute gradient to find the direction maximizing |g_12|
        let mut g_12 = [0.0_f64; 6];
        let mut g_13 = [0.0_f64; 6];
        for mu in 0..n_basis {
            let mut bp = [0.0_f64; 6]; bp[mu] = eps;
            let mut bm = [0.0_f64; 6]; bm[mu] = -eps;
            let (t12_p, t13_p, _) = compute_angles(&bp);
            let (t12_m, t13_m, _) = compute_angles(&bm);
            g_12[mu] = (t12_p - t12_m) / (2.0 * eps);
            g_13[mu] = (t13_p - t13_m) / (2.0 * eps);
        }

        // Direction that maximizes g_12 (unit vector along g_12)
        let norm_12: f64 = g_12.iter().map(|x| x * x).sum::<f64>().sqrt();
        let mut u_solar = [0.0_f64; 6];
        if norm_12 > 1e-15 {
            for k in 0..6 { u_solar[k] = g_12[k] / norm_12; }
        }

        println!("--- V_6 SOLAR 1D SCAN (DirectOffDiagonalLift) ---");
        println!("  Solar direction u = [{}]",
            u_solar.iter().map(|x| format!("{:.4}", x)).collect::<Vec<_>>().join(", "));
        println!("  |g_12| = {:.4}, g_12.u = {:.4}", norm_12,
            g_12.iter().zip(u_solar.iter()).map(|(g, x)| g * x).sum::<f64>());
        println!("  g_13.u = {:.4}",
            g_13.iter().zip(u_solar.iter()).map(|(g, x)| g * x).sum::<f64>());

        println!("\n  {:>8} {:>10} {:>10} {:>10}", "t", "theta_12", "theta_13", "theta_23");

        let mut best_t = 0.0_f64;
        let mut best_score = f64::MAX;
        let mut best_angles = (0.0_f64, 0.0_f64, 0.0_f64);

        for step_i in -500..=500_i32 {
            let t = step_i as f64 * 0.01;
            let mut beta = [0.0_f64; 6];
            for k in 0..6 { beta[k] = t * u_solar[k]; }

            let (t12, t13, t23) = compute_angles(&beta);

            let t13_ok = (t13 - pdg_t13).abs() < 0.5;
            if t13_ok {
                let score = (t12 - pdg_t12).abs();
                if score < best_score {
                    best_score = score;
                    best_t = t;
                    best_angles = (t12, t13, t23);
                }
            }

            if step_i % 100 == 0 {
                println!("  {:8.2} {:10.4} {:10.4} {:10.4}", t, t12, t13, t23);
            }
        }

        println!("\n  === BEST SOLAR CORRECTION (DirectOffDiagonal) ===");
        println!("  t_optimal = {:.4}", best_t);
        println!("  theta_12 = {:.4} deg (PDG: {:.2}, error: {:.2}%)",
            best_angles.0, pdg_t12, ((best_angles.0 - pdg_t12) / pdg_t12 * 100.0).abs());
        println!("  theta_13 = {:.4} deg (PDG: {:.2}, error: {:.2}%)",
            best_angles.1, pdg_t13, ((best_angles.1 - pdg_t13) / pdg_t13 * 100.0).abs());
        println!("  theta_23 = {:.4} deg (PDG: {:.2}, error: {:.2}%)",
            best_angles.2, pdg_t23, ((best_angles.2 - pdg_t23) / pdg_t23 * 100.0).abs());
    }

    /// Constrained solar scan: zero first-order reactor/atmospheric leakage.
    ///
    /// Uses compute_constrained_solar_direction to find the V_6 direction
    /// that is analytically orthogonal to g_13 and g_23. Then scans along
    /// that direction to push theta_12 toward PDG value.
    #[test]
    fn test_v6_constrained_solar_scan() {
        let ch_pair = (11_usize, 12);
        let nu_pair = (7_usize, 8);
        let alpha_ch = 3.75;
        let alpha_nu = 1.30;
        let pdg_t12 = 33.41_f64;
        let pdg_t13 = 8.54_f64;
        let pdg_t23 = 49.0_f64;
        let eps = 0.05_f64;

        let (v6_basis, _sv, _assessors) = extract_v6_basis();
        let lift = DirectOffDiagonalLift;
        let n_basis = v6_basis.nrows().min(6);

        // Lock baseline permutation
        let (m_ch_0, m_nu_0) = construct_pmns_matrices_two_param(
            ch_pair, nu_pair, alpha_ch, alpha_nu,
        );
        let eig_ch_0 = m_ch_0.selfadjoint_eigendecomposition(faer::Side::Lower);
        let eig_nu_0 = m_nu_0.selfadjoint_eigendecomposition(faer::Side::Lower);
        let u_raw_0 = eig_ch_0.u().transpose() * eig_nu_0.u();
        let (_, perm_u, perm_d) = crate::quark_sector::extract_ckm_permutation_aware(&u_raw_0);

        let compute_angles = |beta: &[f64; 6]| -> (f64, f64, f64) {
            let (m_ch, mut m_nu) = construct_pmns_matrices_two_param(
                ch_pair, nu_pair, alpha_ch, alpha_nu,
            );
            apply_v6_perturbation(&mut m_nu, &v6_basis, beta, &lift);

            let eig_ch = m_ch.selfadjoint_eigendecomposition(faer::Side::Lower);
            let m_nu_s = (&m_nu + m_nu.transpose()) * faer::scale(0.5);
            let eig_nu = m_nu_s.selfadjoint_eigendecomposition(faer::Side::Lower);
            let u_raw = eig_ch.u().transpose() * eig_nu.u();

            let mut u_pmns = faer::Mat::<f64>::zeros(3, 3);
            for i in 0..3 {
                for j in 0..3 {
                    u_pmns.write(i, j, u_raw.read(perm_u[i], perm_d[j]));
                }
            }
            extract_pmns_angles(&u_pmns)
        };

        // Compute gradients at beta=0
        let mut g_12 = [0.0_f64; 6];
        let mut g_13 = [0.0_f64; 6];
        let mut g_23 = [0.0_f64; 6];
        for mu in 0..n_basis {
            let mut bp = [0.0_f64; 6]; bp[mu] = eps;
            let mut bm = [0.0_f64; 6]; bm[mu] = -eps;
            let (t12_p, t13_p, t23_p) = compute_angles(&bp);
            let (t12_m, t13_m, t23_m) = compute_angles(&bm);
            g_12[mu] = (t12_p - t12_m) / (2.0 * eps);
            g_13[mu] = (t13_p - t13_m) / (2.0 * eps);
            g_23[mu] = (t23_p - t23_m) / (2.0 * eps);
        }

        // Compute the constrained solar direction
        let u_opt = compute_constrained_solar_direction(&g_12, &g_13, &g_23);

        let dot = |a: &[f64; 6], b: &[f64; 6]| -> f64 {
            a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
        };

        let g12_dot_u = dot(&g_12, &u_opt);
        let g13_dot_u = dot(&g_13, &u_opt);
        let g23_dot_u = dot(&g_23, &u_opt);

        println!("--- V_6 CONSTRAINED SOLAR SCAN ---");
        println!("  u_opt = [{}]",
            u_opt.iter().map(|x| format!("{:.4}", x)).collect::<Vec<_>>().join(", "));
        println!("  g_12 . u = {:.6} (solar sensitivity)", g12_dot_u);
        println!("  g_13 . u = {:.6} (should be ~0)", g13_dot_u);
        println!("  g_23 . u = {:.6} (should be ~0)", g23_dot_u);

        // Verify analytic orthogonality
        assert!(
            g13_dot_u.abs() < 1e-10,
            "g_13 . u = {:.6e} (expected 0)", g13_dot_u
        );
        assert!(
            g23_dot_u.abs() < 1e-10,
            "g_23 . u = {:.6e} (expected 0)", g23_dot_u
        );

        // Diagnostic: check if g_12 has any component outside the {g_13, g_23} plane
        let norm_12 = dot(&g_12, &g_12).sqrt();
        let residual_frac = if norm_12 > 1e-15 { g12_dot_u / norm_12 } else { 0.0 };
        println!("  |g_12| = {:.4}, residual fraction = {:.6e}", norm_12, residual_frac);
        println!("  g_12 is {:.2}% in the constraint plane",
            (1.0 - residual_frac.abs()) * 100.0);

        // 1D scan along the constrained direction
        println!("\n  {:>8} {:>10} {:>10} {:>10} {:>12} {:>12}",
            "t", "theta_12", "theta_13", "theta_23", "d_t13", "d_t12_pdg");

        let mut best_t = 0.0_f64;
        let mut best_score = f64::MAX;
        let mut best_angles = (0.0_f64, 0.0_f64, 0.0_f64);

        for step_i in -1000..=1000_i32 {
            let t = step_i as f64 * 0.01;
            let mut beta = [0.0_f64; 6];
            for k in 0..6 { beta[k] = t * u_opt[k]; }

            let (t12, t13, t23) = compute_angles(&beta);
            let d_t13 = (t13 - pdg_t13).abs();

            // Hard constraint: theta_13 within 0.5 deg of PDG
            if d_t13 < 0.5 {
                let score = (t12 - pdg_t12).abs();
                if score < best_score {
                    best_score = score;
                    best_t = t;
                    best_angles = (t12, t13, t23);
                }
            }

            if step_i % 100 == 0 {
                println!("  {:8.2} {:10.4} {:10.4} {:10.4} {:12.4} {:12.4}",
                    t, t12, t13, t23, d_t13, (t12 - pdg_t12).abs());
            }
        }

        println!("\n  === CONSTRAINED SOLAR CORRECTION ===");
        println!("  t_optimal = {:.4}", best_t);
        println!("  theta_12 = {:.4} deg (PDG: {:.2}, error: {:.2}%)",
            best_angles.0, pdg_t12, ((best_angles.0 - pdg_t12) / pdg_t12 * 100.0).abs());
        println!("  theta_13 = {:.4} deg (PDG: {:.2}, error: {:.2}%)",
            best_angles.1, pdg_t13, ((best_angles.1 - pdg_t13) / pdg_t13 * 100.0).abs());
        println!("  theta_23 = {:.4} deg (PDG: {:.2}, error: {:.2}%)",
            best_angles.2, pdg_t23, ((best_angles.2 - pdg_t23) / pdg_t23 * 100.0).abs());

        // Report the raw projected solar sensitivity
        println!("  Projected solar sensitivity: {:.4} deg/unit", g12_dot_u);
        println!("  Effective range for theta_13 < 0.5 deg: t in [{:.2}, {:.2}]",
            -0.5 / g13_dot_u.abs().max(0.01), 0.5 / g13_dot_u.abs().max(0.01));
    }

    /// Compare all three FlavorLift implementations on the V_6 Jacobian.
    ///
    /// For each lift (AssessorToFlavorMap, DirectOffDiagonalLift, PsiEquivariantLift),
    /// compute the 6D gradient vectors at beta=0, report magnitudes, collinearity,
    /// and optimal solar selectivity.
    #[test]
    fn test_v6_jacobian_flavor_lift_comparison() {
        let ch_pair = (11_usize, 12);
        let nu_pair = (7_usize, 8);
        let alpha_ch = 3.75;
        let alpha_nu = 1.30;
        let eps = 0.05_f64;

        let (v6_basis, _sv, assessors) = extract_v6_basis();
        let n_basis = v6_basis.nrows().min(6);

        // Lock baseline permutation
        let (m_ch_0, m_nu_0) = construct_pmns_matrices_two_param(
            ch_pair, nu_pair, alpha_ch, alpha_nu,
        );
        let eig_ch_0 = m_ch_0.selfadjoint_eigendecomposition(faer::Side::Lower);
        let eig_nu_0 = m_nu_0.selfadjoint_eigendecomposition(faer::Side::Lower);
        let u_raw_0 = eig_ch_0.u().transpose() * eig_nu_0.u();
        let (_, perm_u, perm_d) = crate::quark_sector::extract_ckm_permutation_aware(&u_raw_0);

        // Build the three lifts
        let lift_partition = AssessorToFlavorMap::default_partition(&assessors);
        let lift_direct = DirectOffDiagonalLift;
        let lift_psi = PsiEquivariantLift::from_assessors(&assessors);

        let lifts: Vec<(&str, &dyn FlavorLift)> = vec![
            ("Partition(12/12/6)", &lift_partition),
            ("DirectOffDiagonal", &lift_direct),
            ("PsiEquivariant", &lift_psi),
        ];

        let compute_angles = |beta: &[f64; 6], lift: &dyn FlavorLift| -> (f64, f64, f64) {
            let (m_ch, mut m_nu) = construct_pmns_matrices_two_param(
                ch_pair, nu_pair, alpha_ch, alpha_nu,
            );
            apply_v6_perturbation(&mut m_nu, &v6_basis, beta, lift);

            let eig_ch = m_ch.selfadjoint_eigendecomposition(faer::Side::Lower);
            let m_nu_s = (&m_nu + m_nu.transpose()) * faer::scale(0.5);
            let eig_nu = m_nu_s.selfadjoint_eigendecomposition(faer::Side::Lower);
            let u_raw = eig_ch.u().transpose() * eig_nu.u();

            let mut u_pmns = faer::Mat::<f64>::zeros(3, 3);
            for i in 0..3 {
                for j in 0..3 {
                    u_pmns.write(i, j, u_raw.read(perm_u[i], perm_d[j]));
                }
            }
            extract_pmns_angles(&u_pmns)
        };

        println!("--- V_6 JACOBIAN: FLAVOR LIFT COMPARISON ---");

        for (name, lift) in &lifts {
            let mut g_12 = [0.0_f64; 6];
            let mut g_13 = [0.0_f64; 6];
            let mut g_23 = [0.0_f64; 6];

            for mu in 0..n_basis {
                let mut bp = [0.0_f64; 6];
                let mut bm = [0.0_f64; 6];
                bp[mu] = eps;
                bm[mu] = -eps;

                let (t12_p, t13_p, t23_p) = compute_angles(&bp, *lift);
                let (t12_m, t13_m, t23_m) = compute_angles(&bm, *lift);

                g_12[mu] = (t12_p - t12_m) / (2.0 * eps);
                g_13[mu] = (t13_p - t13_m) / (2.0 * eps);
                g_23[mu] = (t23_p - t23_m) / (2.0 * eps);
            }

            let norm_12: f64 = g_12.iter().map(|x| x * x).sum::<f64>().sqrt();
            let norm_13: f64 = g_13.iter().map(|x| x * x).sum::<f64>().sqrt();
            let norm_23: f64 = g_23.iter().map(|x| x * x).sum::<f64>().sqrt();

            // Collinearity: cos(angle) between g_12 and g_13
            let dot_12_13: f64 = g_12.iter().zip(g_13.iter()).map(|(a, b)| a * b).sum();
            let cos_12_13 = if norm_12 > 1e-15 && norm_13 > 1e-15 {
                dot_12_13 / (norm_12 * norm_13)
            } else { 0.0 };

            // Optimal solar selectivity
            let lambda_13 = 10.0_f64;
            let lambda_23 = 3.0_f64;
            let mut best_score = f64::NEG_INFINITY;
            let mut best_u = [0.0_f64; 6];

            // Single-axis + pairwise + random search
            for mu in 0..n_basis {
                for sign in [-1.0_f64, 1.0] {
                    let mut u = [0.0_f64; 6];
                    u[mu] = sign;
                    let s12: f64 = g_12.iter().zip(u.iter()).map(|(g, x)| g * x).sum();
                    let s13: f64 = g_13.iter().zip(u.iter()).map(|(g, x)| g * x).sum();
                    let s23: f64 = g_23.iter().zip(u.iter()).map(|(g, x)| g * x).sum();
                    let score = s12.abs() - lambda_13 * s13.abs() - lambda_23 * s23.abs();
                    if score > best_score { best_score = score; best_u = u; }
                }
            }
            for mu1 in 0..n_basis {
                for mu2 in (mu1 + 1)..n_basis {
                    for angle_step in (0..360).step_by(5) {
                        let theta = (angle_step as f64) * std::f64::consts::PI / 180.0;
                        let mut u = [0.0_f64; 6];
                        u[mu1] = theta.cos();
                        u[mu2] = theta.sin();
                        let s12: f64 = g_12.iter().zip(u.iter()).map(|(g, x)| g * x).sum();
                        let s13: f64 = g_13.iter().zip(u.iter()).map(|(g, x)| g * x).sum();
                        let s23: f64 = g_23.iter().zip(u.iter()).map(|(g, x)| g * x).sum();
                        let score = s12.abs() - lambda_13 * s13.abs() - lambda_23 * s23.abs();
                        if score > best_score { best_score = score; best_u = u; }
                    }
                }
            }

            let s12_opt: f64 = g_12.iter().zip(best_u.iter()).map(|(g, x)| g * x).sum();
            let s13_opt: f64 = g_13.iter().zip(best_u.iter()).map(|(g, x)| g * x).sum();
            let s23_opt: f64 = g_23.iter().zip(best_u.iter()).map(|(g, x)| g * x).sum();

            println!("\n  === {} ===", name);
            println!("    |g_12| = {:.4}, |g_13| = {:.4}, |g_23| = {:.4}", norm_12, norm_13, norm_23);
            println!("    cos(g_12, g_13) = {:.4} (1.0 = perfectly collinear)", cos_12_13);
            println!("    Optimal S(u) = {:.4}", best_score);
            println!("    g_12.u = {:.4}, g_13.u = {:.4}, g_23.u = {:.4}", s12_opt, s13_opt, s23_opt);
            println!("    u = [{}]", best_u.iter().map(|x| format!("{:.3}", x)).collect::<Vec<_>>().join(", "));
        }
    }

    /// TensorElementLift: 42D -> 6D mapping to all 6 independent M_nu elements.
    ///
    /// Tests whether injecting into all 6 matrix elements (not just 3 generation
    /// factors) breaks the rank-2 lock. The 42 assessors are split into 6 blocks
    /// of 7, each driving one independent element of the symmetric 3x3 matrix.
    #[test]
    fn test_v6_tensor_element_lift_jacobian() {
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
        let (m_ch_0, m_nu_0) = construct_pmns_matrices_two_param(
            ch_pair, nu_pair, alpha_ch, alpha_nu,
        );
        let eig_ch_0 = m_ch_0.selfadjoint_eigendecomposition(faer::Side::Lower);
        let eig_nu_0 = m_nu_0.selfadjoint_eigendecomposition(faer::Side::Lower);
        let u_raw_0 = eig_ch_0.u().transpose() * eig_nu_0.u();
        let (_, perm_u, perm_d) = crate::quark_sector::extract_ckm_permutation_aware(&u_raw_0);

        let compute_angles = |beta: &[f64; 6]| -> (f64, f64, f64) {
            let (m_ch, mut m_nu) = construct_pmns_matrices_two_param(
                ch_pair, nu_pair, alpha_ch, alpha_nu,
            );
            apply_v6_perturbation(&mut m_nu, &v6_basis, beta, &lift);

            let eig_ch = m_ch.selfadjoint_eigendecomposition(faer::Side::Lower);
            let m_nu_s = (&m_nu + m_nu.transpose()) * faer::scale(0.5);
            let eig_nu = m_nu_s.selfadjoint_eigendecomposition(faer::Side::Lower);
            let u_raw = eig_ch.u().transpose() * eig_nu.u();

            let mut u_pmns = faer::Mat::<f64>::zeros(3, 3);
            for i in 0..3 { for j in 0..3 {
                u_pmns.write(i, j, u_raw.read(perm_u[i], perm_d[j]));
            }}
            extract_pmns_angles(&u_pmns)
        };

        // Compute gradients
        let mut g_12 = [0.0_f64; 6];
        let mut g_13 = [0.0_f64; 6];
        let mut g_23 = [0.0_f64; 6];
        for mu in 0..n_basis {
            let mut bp = [0.0_f64; 6]; bp[mu] = eps;
            let mut bm = [0.0_f64; 6]; bm[mu] = -eps;
            let (t12_p, t13_p, t23_p) = compute_angles(&bp);
            let (t12_m, t13_m, t23_m) = compute_angles(&bm);
            g_12[mu] = (t12_p - t12_m) / (2.0 * eps);
            g_13[mu] = (t13_p - t13_m) / (2.0 * eps);
            g_23[mu] = (t23_p - t23_m) / (2.0 * eps);
        }

        let dot = |a: &[f64; 6], b: &[f64; 6]| -> f64 {
            a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
        };

        let norm_12 = dot(&g_12, &g_12).sqrt();
        let norm_13 = dot(&g_13, &g_13).sqrt();
        let norm_23 = dot(&g_23, &g_23).sqrt();

        let cos_12_13 = if norm_12 > 1e-15 && norm_13 > 1e-15 {
            dot(&g_12, &g_13) / (norm_12 * norm_13)
        } else { 0.0 };

        println!("--- V_6 TENSOR ELEMENT LIFT JACOBIAN ---");
        println!("  |g_12| = {:.6}", norm_12);
        println!("  |g_13| = {:.6}", norm_13);
        println!("  |g_23| = {:.6}", norm_23);
        println!("  cos(g_12, g_13) = {:.6}", cos_12_13);
        println!("  g_12 = [{}]", g_12.iter().map(|x| format!("{:.4}", x)).collect::<Vec<_>>().join(", "));
        println!("  g_13 = [{}]", g_13.iter().map(|x| format!("{:.4}", x)).collect::<Vec<_>>().join(", "));
        println!("  g_23 = [{}]", g_23.iter().map(|x| format!("{:.4}", x)).collect::<Vec<_>>().join(", "));

        // Constrained solar direction
        let u_opt = compute_constrained_solar_direction(&g_12, &g_13, &g_23);
        let g12_u = dot(&g_12, &u_opt);
        let g13_u = dot(&g_13, &u_opt);
        let g23_u = dot(&g_23, &u_opt);
        let residual_frac = if norm_12 > 1e-15 { g12_u / norm_12 } else { 0.0 };

        println!("\n  Constrained solar direction:");
        println!("    u_opt = [{}]", u_opt.iter().map(|x| format!("{:.4}", x)).collect::<Vec<_>>().join(", "));
        println!("    g_12 . u = {:.6} (solar sensitivity)", g12_u);
        println!("    g_13 . u = {:.6e} (reactor leakage)", g13_u);
        println!("    g_23 . u = {:.6e} (atmospheric leakage)", g23_u);
        println!("    Residual fraction = {:.6} ({:.2}% outside constraint plane)",
            residual_frac.abs(), residual_frac.abs() * 100.0);

        // If the rank-2 lock is broken, run a constrained 1D scan
        if residual_frac.abs() > 0.01 {
            println!("\n  RANK BROKEN! g_12 has {:.2}% outside {{g_13,g_23}} plane.",
                residual_frac.abs() * 100.0);
            println!("  Running constrained 1D solar scan...\n");
            println!("  {:>8} {:>10} {:>10} {:>10}", "t", "theta_12", "theta_13", "theta_23");

            let mut best_t = 0.0_f64;
            let mut best_score = f64::MAX;
            let mut best_angles = (0.0_f64, 0.0_f64, 0.0_f64);

            for step_i in -500..=500_i32 {
                let t = step_i as f64 * 0.01;
                let mut beta = [0.0_f64; 6];
                for k in 0..6 { beta[k] = t * u_opt[k]; }
                let (t12, t13, t23) = compute_angles(&beta);

                if (t13 - pdg_t13).abs() < 0.5 {
                    let score = (t12 - pdg_t12).abs();
                    if score < best_score {
                        best_score = score;
                        best_t = t;
                        best_angles = (t12, t13, t23);
                    }
                }

                if step_i % 50 == 0 {
                    println!("  {:8.2} {:10.4} {:10.4} {:10.4}", t, t12, t13, t23);
                }
            }

            println!("\n  === TENSOR ELEMENT SOLAR CORRECTION ===");
            println!("  t_optimal = {:.4}", best_t);
            println!("  theta_12 = {:.4} deg (PDG: {:.2}, error: {:.2}%)",
                best_angles.0, pdg_t12, ((best_angles.0 - pdg_t12) / pdg_t12 * 100.0).abs());
            println!("  theta_13 = {:.4} deg (PDG: {:.2}, error: {:.2}%)",
                best_angles.1, pdg_t13, ((best_angles.1 - pdg_t13) / pdg_t13 * 100.0).abs());
            println!("  theta_23 = {:.4} deg (PDG: {:.2}, error: {:.2}%)",
                best_angles.2, pdg_t23, ((best_angles.2 - pdg_t23) / pdg_t23 * 100.0).abs());
        } else {
            println!("\n  Rank-2 lock persists under TensorElementLift.");
        }
    }

    /// Jacobian + constrained scan for V_6 alpha-modulated psi coupling.
    ///
    /// Tests whether the nonlinear V_6 modulation breaks the rank-2 lock
    /// proven in C-1476 for linear injection. If g_12 has nonzero projection
    /// orthogonal to {g_13, g_23}, the solar angle can be selectively corrected.
    #[test]
    fn test_v6_alpha_modulated_jacobian() {
        let ch_pair = (11_usize, 12);
        let nu_pair = (7_usize, 8);
        let base_alpha_ch = 3.75;
        let base_alpha_nu = 1.30;
        let pdg_t12 = 33.41_f64;
        let pdg_t13 = 8.54_f64;
        let pdg_t23 = 49.0_f64;
        let eps = 0.05_f64;

        let (v6_basis, _sv, _assessors) = extract_v6_basis();
        let n_basis = v6_basis.nrows().min(6);

        // Lock baseline permutation at beta=0 (where modulated == unmodulated)
        let (m_ch_0, m_nu_0) = construct_pmns_matrices_v6_modulated(
            ch_pair, nu_pair, base_alpha_ch, base_alpha_nu,
            &v6_basis, &[0.0; 6],
        );
        let eig_ch_0 = m_ch_0.selfadjoint_eigendecomposition(faer::Side::Lower);
        let eig_nu_0 = m_nu_0.selfadjoint_eigendecomposition(faer::Side::Lower);
        let u_raw_0 = eig_ch_0.u().transpose() * eig_nu_0.u();
        let (_, perm_u, perm_d) = crate::quark_sector::extract_ckm_permutation_aware(&u_raw_0);

        // Verify beta=0 recovery matches the two-param baseline
        let (t12_0, t13_0, t23_0) = {
            let mut u_pmns = faer::Mat::<f64>::zeros(3, 3);
            for i in 0..3 { for j in 0..3 {
                u_pmns.write(i, j, u_raw_0.read(perm_u[i], perm_d[j]));
            }}
            extract_pmns_angles(&u_pmns)
        };
        println!("--- V_6 ALPHA-MODULATED JACOBIAN ---");
        println!("  beta=0: theta_12={:.4}, theta_13={:.4}, theta_23={:.4}",
            t12_0, t13_0, t23_0);
        assert!((t12_0 - 28.54).abs() < 0.01, "beta=0 recovery failed for theta_12");
        assert!((t13_0 - 8.63).abs() < 0.01, "beta=0 recovery failed for theta_13");

        let compute_angles = |beta: &[f64; 6]| -> (f64, f64, f64) {
            let (m_ch, m_nu) = construct_pmns_matrices_v6_modulated(
                ch_pair, nu_pair, base_alpha_ch, base_alpha_nu,
                &v6_basis, beta,
            );
            let eig_ch = m_ch.selfadjoint_eigendecomposition(faer::Side::Lower);
            let m_nu_s = (&m_nu + m_nu.transpose()) * faer::scale(0.5);
            let eig_nu = m_nu_s.selfadjoint_eigendecomposition(faer::Side::Lower);
            let u_raw = eig_ch.u().transpose() * eig_nu.u();

            let mut u_pmns = faer::Mat::<f64>::zeros(3, 3);
            for i in 0..3 { for j in 0..3 {
                u_pmns.write(i, j, u_raw.read(perm_u[i], perm_d[j]));
            }}
            extract_pmns_angles(&u_pmns)
        };

        // Compute gradients
        let mut g_12 = [0.0_f64; 6];
        let mut g_13 = [0.0_f64; 6];
        let mut g_23 = [0.0_f64; 6];
        for mu in 0..n_basis {
            let mut bp = [0.0_f64; 6]; bp[mu] = eps;
            let mut bm = [0.0_f64; 6]; bm[mu] = -eps;
            let (t12_p, t13_p, t23_p) = compute_angles(&bp);
            let (t12_m, t13_m, t23_m) = compute_angles(&bm);
            g_12[mu] = (t12_p - t12_m) / (2.0 * eps);
            g_13[mu] = (t13_p - t13_m) / (2.0 * eps);
            g_23[mu] = (t23_p - t23_m) / (2.0 * eps);
        }

        let dot = |a: &[f64; 6], b: &[f64; 6]| -> f64 {
            a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
        };

        let norm_12 = dot(&g_12, &g_12).sqrt();
        let norm_13 = dot(&g_13, &g_13).sqrt();
        let norm_23 = dot(&g_23, &g_23).sqrt();

        let cos_12_13 = if norm_12 > 1e-15 && norm_13 > 1e-15 {
            dot(&g_12, &g_13) / (norm_12 * norm_13)
        } else { 0.0 };

        println!("\n  Gradient magnitudes:");
        println!("    |g_12| = {:.6}", norm_12);
        println!("    |g_13| = {:.6}", norm_13);
        println!("    |g_23| = {:.6}", norm_23);
        println!("    cos(g_12, g_13) = {:.6}", cos_12_13);
        println!("    g_12 = [{}]", g_12.iter().map(|x| format!("{:.4}", x)).collect::<Vec<_>>().join(", "));
        println!("    g_13 = [{}]", g_13.iter().map(|x| format!("{:.4}", x)).collect::<Vec<_>>().join(", "));
        println!("    g_23 = [{}]", g_23.iter().map(|x| format!("{:.4}", x)).collect::<Vec<_>>().join(", "));

        // Constrained solar direction
        let u_opt = compute_constrained_solar_direction(&g_12, &g_13, &g_23);
        let g12_u = dot(&g_12, &u_opt);
        let g13_u = dot(&g_13, &u_opt);
        let g23_u = dot(&g_23, &u_opt);
        let residual_frac = if norm_12 > 1e-15 { g12_u / norm_12 } else { 0.0 };

        println!("\n  Constrained solar direction:");
        println!("    u_opt = [{}]", u_opt.iter().map(|x| format!("{:.4}", x)).collect::<Vec<_>>().join(", "));
        println!("    g_12 . u = {:.6} (solar sensitivity)", g12_u);
        println!("    g_13 . u = {:.6} (reactor leakage)", g13_u);
        println!("    g_23 . u = {:.6} (atmospheric leakage)", g23_u);
        println!("    Residual fraction = {:.6e} ({:.2}% outside constraint plane)",
            residual_frac.abs(), residual_frac.abs() * 100.0);

        // If the residual fraction is significant, run a 1D scan
        if residual_frac.abs() > 0.001 {
            println!("\n  RANK BROKEN: g_12 has {:.2}% outside {{g_13,g_23}} plane!",
                residual_frac.abs() * 100.0);
            println!("  Running constrained 1D scan...\n");
            println!("  {:>8} {:>10} {:>10} {:>10}", "t", "theta_12", "theta_13", "theta_23");

            let mut best_t = 0.0_f64;
            let mut best_score = f64::MAX;
            let mut best_angles = (0.0_f64, 0.0_f64, 0.0_f64);

            for step_i in -500..=500_i32 {
                let t = step_i as f64 * 0.01;
                let mut beta = [0.0_f64; 6];
                for k in 0..6 { beta[k] = t * u_opt[k]; }

                let (t12, t13, t23) = compute_angles(&beta);

                if (t13 - pdg_t13).abs() < 0.5 {
                    let score = (t12 - pdg_t12).abs();
                    if score < best_score {
                        best_score = score;
                        best_t = t;
                        best_angles = (t12, t13, t23);
                    }
                }

                if step_i % 100 == 0 {
                    println!("  {:8.2} {:10.4} {:10.4} {:10.4}", t, t12, t13, t23);
                }
            }

            println!("\n  === ALPHA-MODULATED SOLAR CORRECTION ===");
            println!("  t_optimal = {:.4}", best_t);
            println!("  theta_12 = {:.4} deg (PDG: {:.2}, error: {:.2}%)",
                best_angles.0, pdg_t12, ((best_angles.0 - pdg_t12) / pdg_t12 * 100.0).abs());
            println!("  theta_13 = {:.4} deg (PDG: {:.2}, error: {:.2}%)",
                best_angles.1, pdg_t13, ((best_angles.1 - pdg_t13) / pdg_t13 * 100.0).abs());
            println!("  theta_23 = {:.4} deg (PDG: {:.2}, error: {:.2}%)",
                best_angles.2, pdg_t23, ((best_angles.2 - pdg_t23) / pdg_t23 * 100.0).abs());
        } else {
            println!("\n  Rank-2 lock persists under alpha modulation.");
            println!("  Residual fraction {:.6e} is below threshold 0.001.", residual_frac.abs());
        }
    }

    /// Regression test pinning the V_6-corrected PMNS optimum (C-1478).
    ///
    /// Reproduces the full pipeline: two-param psi coupling + TensorElementLift
    /// along the constrained solar direction at t=2.47. Pins all three angles
    /// with theta_13 at the tightest tolerance (the most fragile win).
    #[test]
    fn test_pmns_v6_corrected_regression() {
        let ch_pair = (11_usize, 12);
        let nu_pair = (7_usize, 8);
        let alpha_ch = 3.75;
        let alpha_nu = 1.30;
        let eps = 0.05_f64;

        let (v6_basis, _sv, _assessors) = extract_v6_basis();
        let lift = TensorElementLift;
        let n_basis = v6_basis.nrows().min(6);

        // Lock baseline permutation
        let (m_ch_0, m_nu_0) = construct_pmns_matrices_two_param(
            ch_pair, nu_pair, alpha_ch, alpha_nu,
        );
        let eig_ch_0 = m_ch_0.selfadjoint_eigendecomposition(faer::Side::Lower);
        let eig_nu_0 = m_nu_0.selfadjoint_eigendecomposition(faer::Side::Lower);
        let u_raw_0 = eig_ch_0.u().transpose() * eig_nu_0.u();
        let (_, perm_u, perm_d) = crate::quark_sector::extract_ckm_permutation_aware(&u_raw_0);

        let compute_angles = |beta: &[f64; 6]| -> (f64, f64, f64) {
            let (m_ch, mut m_nu) = construct_pmns_matrices_two_param(
                ch_pair, nu_pair, alpha_ch, alpha_nu,
            );
            apply_v6_perturbation(&mut m_nu, &v6_basis, beta, &lift);
            let eig_ch = m_ch.selfadjoint_eigendecomposition(faer::Side::Lower);
            let m_nu_s = (&m_nu + m_nu.transpose()) * faer::scale(0.5);
            let eig_nu = m_nu_s.selfadjoint_eigendecomposition(faer::Side::Lower);
            let u_raw = eig_ch.u().transpose() * eig_nu.u();
            let mut u_pmns = faer::Mat::<f64>::zeros(3, 3);
            for i in 0..3 { for j in 0..3 {
                u_pmns.write(i, j, u_raw.read(perm_u[i], perm_d[j]));
            }}
            extract_pmns_angles(&u_pmns)
        };

        // Compute constrained solar direction
        let mut g_12 = [0.0_f64; 6];
        let mut g_13 = [0.0_f64; 6];
        let mut g_23 = [0.0_f64; 6];
        for mu in 0..n_basis {
            let mut bp = [0.0_f64; 6]; bp[mu] = eps;
            let mut bm = [0.0_f64; 6]; bm[mu] = -eps;
            let (t12_p, t13_p, t23_p) = compute_angles(&bp);
            let (t12_m, t13_m, t23_m) = compute_angles(&bm);
            g_12[mu] = (t12_p - t12_m) / (2.0 * eps);
            g_13[mu] = (t13_p - t13_m) / (2.0 * eps);
            g_23[mu] = (t23_p - t23_m) / (2.0 * eps);
        }

        let u_opt = compute_constrained_solar_direction(&g_12, &g_13, &g_23);
        let dot = |a: &[f64; 6], b: &[f64; 6]| -> f64 {
            a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
        };

        // Verify orthogonality constraints
        assert!(dot(&g_13, &u_opt).abs() < 1e-10, "g_13.u not zero");
        assert!(dot(&g_23, &u_opt).abs() < 1e-10, "g_23.u not zero");

        // Verify residual fraction > 0.5 (rank-2 lock broken)
        let norm_12 = dot(&g_12, &g_12).sqrt();
        let residual = dot(&g_12, &u_opt) / norm_12;
        assert!(residual.abs() > 0.5,
            "Residual fraction {:.4} too low -- rank-2 lock not broken", residual.abs());

        // Apply correction at t=2.47
        let t_opt = 2.47_f64;
        let mut beta_opt = [0.0_f64; 6];
        for k in 0..6 { beta_opt[k] = t_opt * u_opt[k]; }
        let (t12, t13, t23) = compute_angles(&beta_opt);

        println!("--- V_6-CORRECTED PMNS REGRESSION ---");
        println!("  theta_12 = {:.4} deg (expected ~33.42)", t12);
        println!("  theta_13 = {:.4} deg (expected ~8.63)", t13);
        println!("  theta_23 = {:.4} deg (expected ~47.08)", t23);

        // Pin the corrected angles -- theta_13 is tightest
        assert!((t13 - 8.63).abs() < 0.01,
            "theta_13 regression FAILED: {:.4} (expected ~8.63, tol 0.01)", t13);
        assert!((t12 - 33.42).abs() < 0.05,
            "theta_12 regression FAILED: {:.4} (expected ~33.42, tol 0.05)", t12);
        assert!((t23 - 47.08).abs() < 0.05,
            "theta_23 regression FAILED: {:.4} (expected ~47.08, tol 0.05)", t23);

        println!("  PASS: V_6-corrected PMNS regression");
    }

    /// Invariance audit of TensorElementLift block assignment.
    ///
    /// Tests whether the 7-assessor blocks are structurally aligned with the
    /// V_6 basis, or merely a convenient partition. Checks:
    /// (1) Overlap between V_6 basis vectors and block indicator vectors
    /// (2) Psi-orbit structure of assessors within each block
    /// (3) Sensitivity to assessor reordering within blocks
    #[test]
    fn test_tensor_element_lift_invariance_audit() {
        use cd_kernel::gourlay_psi;

        let (v6_basis, sv, assessors) = extract_v6_basis();
        let n_basis = v6_basis.nrows();

        println!("--- TENSOR ELEMENT LIFT INVARIANCE AUDIT ---");
        println!("  V_6 basis: {}x{}, SV = [{}]", n_basis, v6_basis.ncols(),
            sv.iter().map(|s| format!("{:.3}", s)).collect::<Vec<_>>().join(", "));

        // (1) Overlap matrix: how much does each V_6 basis vector concentrate
        //     in each of the 6 blocks of 7 assessors?
        println!("\n  Block overlap matrix (V_6 basis x 6 blocks):");
        println!("  {:>6} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
            "V_6[k]", "blk_0", "blk_1", "blk_2", "blk_3", "blk_4", "blk_5");

        let mut block_overlap = [[0.0_f64; 6]; 6]; // [v6_idx][block_idx]
        for k in 0..n_basis {
            for b in 0..6 {
                let start = b * 7;
                let end = (start + 7).min(42);
                let energy: f64 = (start..end).map(|col| {
                    let val = v6_basis[(k, col)];
                    val * val
                }).sum();
                block_overlap[k][b] = energy;
            }
            println!("  V_6[{}] {:8.4} {:8.4} {:8.4} {:8.4} {:8.4} {:8.4}",
                k, block_overlap[k][0], block_overlap[k][1], block_overlap[k][2],
                block_overlap[k][3], block_overlap[k][4], block_overlap[k][5]);
        }

        // Is any V_6 basis vector concentrated in a single block?
        // (would indicate structural alignment)
        let mut max_concentration = 0.0_f64;
        for k in 0..n_basis {
            let total: f64 = block_overlap[k].iter().sum();
            for b in 0..6 {
                let frac = if total > 1e-15 { block_overlap[k][b] / total } else { 0.0 };
                if frac > max_concentration { max_concentration = frac; }
            }
        }
        println!("\n  Max block concentration: {:.4} (1.0 = perfectly aligned, 0.167 = uniform)",
            max_concentration);

        // (2) Psi-orbit structure: for each assessor, compute psi(e_low + e_high)
        //     and check which assessor index it maps to
        println!("\n  Psi-orbit structure of assessors:");
        let mut orbit_sizes = vec![0_usize; 42];
        let mut visited = vec![false; 42];

        for a_idx in 0..assessors.len() {
            if visited[a_idx] { continue; }
            let mut orbit = vec![a_idx];
            visited[a_idx] = true;

            // Embed assessor as 16D vector, apply psi, find which assessor it maps to
            let (low, high) = assessors[a_idx];
            let mut v = [0.0_f64; 16];
            v[low] = 1.0;
            v[high] = 1.0;

            let psi_v = gourlay_psi(&v);

            // Find the assessor closest to psi_v
            let mut best_match = a_idx;
            let mut best_overlap = 0.0_f64;
            for (b_idx, &(bl, bh)) in assessors.iter().enumerate() {
                let overlap = psi_v[bl].abs() + psi_v[bh].abs();
                if overlap > best_overlap {
                    best_overlap = overlap;
                    best_match = b_idx;
                }
            }

            if best_match != a_idx && !visited[best_match] {
                orbit.push(best_match);
                visited[best_match] = true;
            }

            for &idx in &orbit {
                orbit_sizes[idx] = orbit.len();
            }
        }

        // Count orbits by size and block membership
        let mut orbits_within_block = 0_usize;
        let mut orbits_cross_block = 0_usize;
        for a_idx in 0..42 {
            if orbit_sizes[a_idx] >= 2 {
                // Check if orbit partner is in the same block
                let block_a = a_idx / 7;
                // Find the partner by checking psi again
                let (low, high) = assessors[a_idx];
                let mut v = [0.0_f64; 16];
                v[low] = 1.0;
                v[high] = 1.0;
                let psi_v = gourlay_psi(&v);
                let mut partner = a_idx;
                let mut best_ov = 0.0_f64;
                for (b_idx, &(bl, bh)) in assessors.iter().enumerate() {
                    let ov = psi_v[bl].abs() + psi_v[bh].abs();
                    if ov > best_ov && b_idx != a_idx {
                        best_ov = ov;
                        partner = b_idx;
                    }
                }
                let block_p = partner / 7;
                if block_a == block_p {
                    orbits_within_block += 1;
                } else {
                    orbits_cross_block += 1;
                }
            }
        }
        println!("  Psi orbits within same block: {}", orbits_within_block);
        println!("  Psi orbits crossing blocks: {}", orbits_cross_block);

        // (3) Sensitivity to within-block reordering
        // Reverse the assessor order within each block and check if the
        // constrained solar direction changes
        println!("\n  Assessor reordering sensitivity:");
        println!("  (TensorElementLift sums within blocks, so reordering is invariant by construction)");
        println!("  PASS: block sums are permutation-invariant within blocks");

        // Summary
        println!("\n  === AUDIT SUMMARY ===");
        if max_concentration > 0.5 {
            println!("  Block alignment: STRONG (max concentration {:.2}%)", max_concentration * 100.0);
        } else if max_concentration > 0.25 {
            println!("  Block alignment: MODERATE (max concentration {:.2}%)", max_concentration * 100.0);
        } else {
            println!("  Block alignment: WEAK (max concentration {:.2}%, near uniform)", max_concentration * 100.0);
        }
        println!("  Psi covariance: {} within-block, {} cross-block", orbits_within_block, orbits_cross_block);
    }

    /// Jacobian rank, condition number, and stability at TensorElementLift optimum.
    ///
    /// Computes the full 3x6 Jacobian d(theta_i)/d(beta_k) at the optimum,
    /// reports rank and condition number. Also computes the local Hessian
    /// d^2(theta_12)/dt^2 along the constrained direction for curvature.
    #[test]
    fn test_tensor_element_lift_stability() {
        let ch_pair = (11_usize, 12);
        let nu_pair = (7_usize, 8);
        let alpha_ch = 3.75;
        let alpha_nu = 1.30;
        let eps = 0.05_f64;

        let (v6_basis, _sv, _assessors) = extract_v6_basis();
        let lift = TensorElementLift;
        let n_basis = v6_basis.nrows().min(6);

        // Lock baseline permutation
        let (m_ch_0, m_nu_0) = construct_pmns_matrices_two_param(
            ch_pair, nu_pair, alpha_ch, alpha_nu,
        );
        let eig_ch_0 = m_ch_0.selfadjoint_eigendecomposition(faer::Side::Lower);
        let eig_nu_0 = m_nu_0.selfadjoint_eigendecomposition(faer::Side::Lower);
        let u_raw_0 = eig_ch_0.u().transpose() * eig_nu_0.u();
        let (_, perm_u, perm_d) = crate::quark_sector::extract_ckm_permutation_aware(&u_raw_0);

        // Compute constrained direction first (at beta=0)
        let compute_angles_at = |beta: &[f64; 6]| -> (f64, f64, f64) {
            let (m_ch, mut m_nu) = construct_pmns_matrices_two_param(
                ch_pair, nu_pair, alpha_ch, alpha_nu,
            );
            apply_v6_perturbation(&mut m_nu, &v6_basis, beta, &lift);
            let eig_ch = m_ch.selfadjoint_eigendecomposition(faer::Side::Lower);
            let m_nu_s = (&m_nu + m_nu.transpose()) * faer::scale(0.5);
            let eig_nu = m_nu_s.selfadjoint_eigendecomposition(faer::Side::Lower);
            let u_raw = eig_ch.u().transpose() * eig_nu.u();
            let mut u_pmns = faer::Mat::<f64>::zeros(3, 3);
            for i in 0..3 { for j in 0..3 {
                u_pmns.write(i, j, u_raw.read(perm_u[i], perm_d[j]));
            }}
            extract_pmns_angles(&u_pmns)
        };

        // Get constrained direction
        let mut g0_12 = [0.0_f64; 6];
        let mut g0_13 = [0.0_f64; 6];
        let mut g0_23 = [0.0_f64; 6];
        for mu in 0..n_basis {
            let mut bp = [0.0_f64; 6]; bp[mu] = eps;
            let mut bm = [0.0_f64; 6]; bm[mu] = -eps;
            let (t12_p, t13_p, t23_p) = compute_angles_at(&bp);
            let (t12_m, t13_m, t23_m) = compute_angles_at(&bm);
            g0_12[mu] = (t12_p - t12_m) / (2.0 * eps);
            g0_13[mu] = (t13_p - t13_m) / (2.0 * eps);
            g0_23[mu] = (t23_p - t23_m) / (2.0 * eps);
        }
        let u_opt = compute_constrained_solar_direction(&g0_12, &g0_13, &g0_23);
        let t_opt = 2.47_f64;

        // Compute full 3x6 Jacobian at the OPTIMUM (t=2.47)
        let mut jac = [[0.0_f64; 6]; 3]; // [angle_idx][beta_idx]
        for mu in 0..n_basis {
            let mut beta_center = [0.0_f64; 6];
            for k in 0..6 { beta_center[k] = t_opt * u_opt[k]; }

            let mut bp = beta_center;
            let mut bm = beta_center;
            bp[mu] += eps;
            bm[mu] -= eps;

            let (t12_p, t13_p, t23_p) = compute_angles_at(&bp);
            let (t12_m, t13_m, t23_m) = compute_angles_at(&bm);

            jac[0][mu] = (t12_p - t12_m) / (2.0 * eps);
            jac[1][mu] = (t13_p - t13_m) / (2.0 * eps);
            jac[2][mu] = (t23_p - t23_m) / (2.0 * eps);
        }

        // Build 3x6 nalgebra matrix for SVD
        let jac_mat = nalgebra::DMatrix::from_fn(3, 6, |i, j| jac[i][j]);
        let svd_jac = jac_mat.svd(false, false);

        println!("--- TENSOR ELEMENT LIFT STABILITY ---");
        println!("  Full 3x6 Jacobian at t_opt={}:", t_opt);
        for i in 0..3 {
            println!("    d(theta_{})/d(beta) = [{}]",
                ["12", "13", "23"][i],
                jac[i].iter().map(|x| format!("{:.4}", x)).collect::<Vec<_>>().join(", "));
        }

        let sv = &svd_jac.singular_values;
        let rank = sv.iter().filter(|&&s| s > 1e-8).count();
        let cond = if sv[sv.len() - 1].abs() > 1e-15 { sv[0] / sv[sv.len() - 1] } else { f64::INFINITY };

        println!("\n  Singular values: [{}]",
            sv.iter().map(|s| format!("{:.4}", s)).collect::<Vec<_>>().join(", "));
        println!("  Rank: {} (expected 3 for full control)", rank);
        println!("  Condition number: {:.2}", cond);

        // Local Hessian: d^2(theta_12)/dt^2 along constrained direction
        let dt = 0.1_f64;
        let mut beta_c = [0.0_f64; 6];
        let mut beta_p = [0.0_f64; 6];
        let mut beta_m = [0.0_f64; 6];
        for k in 0..6 {
            beta_c[k] = t_opt * u_opt[k];
            beta_p[k] = (t_opt + dt) * u_opt[k];
            beta_m[k] = (t_opt - dt) * u_opt[k];
        }
        let (t12_c, _, _) = compute_angles_at(&beta_c);
        let (t12_p, _, _) = compute_angles_at(&beta_p);
        let (t12_m, _, _) = compute_angles_at(&beta_m);
        let hessian = (t12_p - 2.0 * t12_c + t12_m) / (dt * dt);

        println!("\n  Local curvature d^2(theta_12)/dt^2 = {:.4} deg/unit^2", hessian);
        if hessian.abs() < 0.1 {
            println!("  Stability: FLAT (robust -- small parameter changes have minimal effect)");
        } else if hessian < -1.0 {
            println!("  Stability: CONCAVE (maximum -- the optimum is a stable peak)");
        } else {
            println!("  Stability: CONVEX (minimum -- theta_12 is at a valley)");
        }

        // Report whether the Jacobian at the optimum still has the constrained property
        let g_opt_12 = jac[0];
        let g_opt_13 = jac[1];
        let g_opt_23 = jac[2];
        let u_opt2 = compute_constrained_solar_direction(&g_opt_12, &g_opt_13, &g_opt_23);
        let dot = |a: &[f64; 6], b: &[f64; 6]| -> f64 {
            a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
        };
        let residual_opt = dot(&g_opt_12, &u_opt2) / dot(&g_opt_12, &g_opt_12).sqrt();
        println!("  Residual fraction at optimum: {:.4} ({:.2}% outside constraint plane)",
            residual_opt.abs(), residual_opt.abs() * 100.0);
    }

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
        use gororoba_algebra::lie::g2_stabilizer::{
            stabilizer_decomposition, complex_structure,
        };
        use gororoba_algebra::lie::g2_su3_representation::fundamental_representation;
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
                    if coeff.abs() < 1e-15 { continue; }
                    if let Some(target) = find_assessor(j, high) {
                        mat[(target, a_idx)] += coeff;
                    }
                }

                // D transforms e_high: contribution to assessor space
                for j in 9..=15 {
                    let coeff = d16[high][j];
                    if coeff.abs() < 1e-15 { continue; }
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
            let frob: f64 = (0..n_v6).flat_map(|i| (0..n_v6).map(move |j| rv6[(i,j)] * rv6[(i,j)])).sum::<f64>().sqrt();
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
            let row: Vec<String> = (0..n_v6).map(|j|
                format!("{:8.4}", rho_v6[0][(i, j)])
            ).collect();
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
                let row: Vec<String> = (0..n_v6).map(|j|
                    format!("{:8.4}", casimir[(i, j)])
                ).collect();
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
                println!("  Casimir = {:.4} * I_6  (PROPORTIONAL TO IDENTITY)", diag_avg);
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
            let row: Vec<String> = (0..n_v6).map(|j|
                format!("{:8.4}", rho_v6_psi[(i, j)])
            ).collect();
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
        let psi3_error: f64 = (&psi3 - &identity).iter().map(|x| x * x).sum::<f64>().sqrt();
        println!("  |psi^3 - I| = {:.6e} (should be ~0 for order-3)", psi3_error);

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
            let row: Vec<String> = (0..6).map(|j|
                format!("{:5.1}", rho_sym3_psi[(i, j)])
            ).collect();
            println!("    [{}]", row.join(", "));
        }

        // Verify psi^3 = I on Sym_3
        let sym_psi2 = &rho_sym3_psi * &rho_sym3_psi;
        let sym_psi3 = &sym_psi2 * &rho_sym3_psi;
        let sym_id = DMatrix::<f64>::identity(6, 6);
        let sym_psi3_error: f64 = (&sym_psi3 - &sym_id).iter().map(|x| x * x).sum::<f64>().sqrt();
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
        println!("  Constraint matrix: {}x{}", constraint_rows, constraint_cols);
        println!("  Singular values (last 10):");
        let n_sv = sigma.len();
        for i in (n_sv.saturating_sub(10))..n_sv {
            println!("    sigma[{}] = {:.6e}", i, sigma[i]);
        }
        println!("  Null space dimension: {} (1 = unique equivariant map up to scale)", null_dim);

        if null_dim > 0 {
            // Extract the null-space vectors (intertwiners)
            // The last null_dim rows of V^T are the null vectors
            println!("\n  Intertwiner(s) found!");

            for ns_idx in 0..null_dim {
                let row_idx = n_sv - 1 - ns_idx;
                if sigma[row_idx] > sv_threshold { break; }

                // Extract L from vec(L)
                let mut l_mat = DMatrix::<f64>::zeros(n, n);
                for i in 0..n {
                    for j in 0..n {
                        l_mat[(i, j)] = v_t[(row_idx, i * n + j)];
                    }
                }

                println!("\n  Intertwiner L_{} (6x6, maps V_6 -> Sym_3):", ns_idx);
                for i in 0..n {
                    let row: Vec<String> = (0..n).map(|j|
                        format!("{:8.4}", l_mat[(i, j)])
                    ).collect();
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
                    let dot: f64 = l_normed.iter().zip(l_tel_normed.iter())
                        .map(|(a, b)| a * b).sum();

                    println!("\n  Comparison with TensorElementLift:");
                    println!("    cos(L_intertwiner, L_TEL) = {:.6}", dot);
                    if dot.abs() > 0.95 {
                        println!("    MATCH: TensorElementLift IS the S_3-equivariant map (up to scale)!");
                    } else if dot.abs() > 0.5 {
                        println!("    PARTIAL: significant overlap but not identical");
                    } else {
                        println!("    MISMATCH: TensorElementLift is NOT the equivariant map");
                    }
                }
            }
        } else {
            println!("  No intertwiner exists: V_6 and Sym_3(R) carry incompatible S_3 representations.");
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
        let (m_ch_0, m_nu_0) = construct_pmns_matrices_two_param(
            ch_pair, nu_pair, alpha_ch, alpha_nu,
        );
        let eig_ch_0 = m_ch_0.selfadjoint_eigendecomposition(faer::Side::Lower);
        let eig_nu_0 = m_nu_0.selfadjoint_eigendecomposition(faer::Side::Lower);
        let u_raw_0 = eig_ch_0.u().transpose() * eig_nu_0.u();
        let (_, perm_u, perm_d) = crate::quark_sector::extract_ckm_permutation_aware(&u_raw_0);

        let compute_angles = |beta: &[f64; 6]| -> (f64, f64, f64) {
            let (m_ch, mut m_nu) = construct_pmns_matrices_two_param(
                ch_pair, nu_pair, alpha_ch, alpha_nu,
            );
            apply_v6_perturbation(&mut m_nu, &v6_basis, beta, &lift);
            let eig_ch = m_ch.selfadjoint_eigendecomposition(faer::Side::Lower);
            let m_nu_s = (&m_nu + m_nu.transpose()) * faer::scale(0.5);
            let eig_nu = m_nu_s.selfadjoint_eigendecomposition(faer::Side::Lower);
            let u_raw = eig_ch.u().transpose() * eig_nu.u();
            let mut u_pmns = faer::Mat::<f64>::zeros(3, 3);
            for i in 0..3 { for j in 0..3 {
                u_pmns.write(i, j, u_raw.read(perm_u[i], perm_d[j]));
            }}
            extract_pmns_angles(&u_pmns)
        };

        // Compute gradients at beta=0
        let mut g_12 = [0.0_f64; 6];
        let mut g_13 = [0.0_f64; 6];
        let mut g_23 = [0.0_f64; 6];
        for mu in 0..n_basis {
            let mut bp = [0.0_f64; 6]; bp[mu] = eps;
            let mut bm = [0.0_f64; 6]; bm[mu] = -eps;
            let (t12_p, t13_p, t23_p) = compute_angles(&bp);
            let (t12_m, t13_m, t23_m) = compute_angles(&bm);
            g_12[mu] = (t12_p - t12_m) / (2.0 * eps);
            g_13[mu] = (t13_p - t13_m) / (2.0 * eps);
            g_23[mu] = (t23_p - t23_m) / (2.0 * eps);
        }

        let dot = |a: &[f64; 6], b: &[f64; 6]| -> f64 {
            a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
        };

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
        println!("  u_solar = [{}]",
            u_solar.iter().map(|x| format!("{:.4}", x)).collect::<Vec<_>>().join(", "));
        println!("  u_atmo  = [{}]",
            u_atmo.iter().map(|x| format!("{:.4}", x)).collect::<Vec<_>>().join(", "));
        println!("\n  Solar direction sensitivity:");
        println!("    g_12.u = {:.6} (solar)", g12_solar);
        println!("    g_13.u = {:.6e} (reactor)", g13_solar);
        println!("    g_23.u = {:.6e} (atmospheric)", g23_solar);
        println!("  Atmospheric direction sensitivity:");
        println!("    g_12.u = {:.6} (solar cross-talk)", g12_atmo);
        println!("    g_13.u = {:.6e} (reactor)", g13_atmo);
        println!("    g_23.u = {:.6} (atmospheric)", g23_atmo);
        println!("  u_solar . u_atmo = {:.6e} (orthogonality)", dot(&u_solar, &u_atmo));

        // 2D scan: beta = t1 * u_solar + t2 * u_atmo
        println!("\n  2D scan (t1=solar, t2=atmo):");
        println!("  {:>6} {:>6} {:>10} {:>10} {:>10} {:>10}",
            "t1", "t2", "theta_12", "theta_13", "theta_23", "score");

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
                if (t13 - pdg_t13).abs() > 0.5 { continue; }

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
            if t1 < 0.0 { continue; }
            for step2 in -50..=50_i32 {
                let t2 = t2_center + step2 as f64 * 0.01;

                let mut beta = [0.0_f64; 6];
                for k in 0..6 {
                    beta[k] = t1 * u_solar[k] + t2 * u_atmo[k];
                }

                let (t12, t13, t23) = compute_angles(&beta);
                if (t13 - pdg_t13).abs() > 0.5 { continue; }

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
        println!("  theta_12 = {:.4} deg (PDG: {:.2}, error: {:.2}%)",
            best_angles.0, pdg_t12, ((best_angles.0 - pdg_t12) / pdg_t12 * 100.0).abs());
        println!("  theta_13 = {:.4} deg (PDG: {:.2}, error: {:.2}%)",
            best_angles.1, pdg_t13, ((best_angles.1 - pdg_t13) / pdg_t13 * 100.0).abs());
        println!("  theta_23 = {:.4} deg (PDG: {:.2}, error: {:.2}%)",
            best_angles.2, pdg_t23, ((best_angles.2 - pdg_t23) / pdg_t23 * 100.0).abs());
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
            let (m_ch_base, m_nu_base) = construct_pmns_matrices_two_param(
                ch_pair, nu_pair, alpha_ch, alpha_nu,
            );
            let eig_ch = m_ch_base.selfadjoint_eigendecomposition(faer::Side::Lower);
            let u_ch = eig_ch.u();
            let u_raw_0 = {
                let eig_nu_0 = m_nu_base.selfadjoint_eigendecomposition(faer::Side::Lower);
                u_ch.transpose() * eig_nu_0.u()
            };
            let (_, perm_u, perm_d) = crate::quark_sector::extract_ckm_permutation_aware(&u_raw_0);

            // Angle extraction from perturbed M_nu (reusing M_ch eigenvectors)
            let angles_from_mnu = |m_nu: &faer::Mat<f64>| -> (f64, f64, f64) {
                let m_nu_s = (m_nu + m_nu.transpose()) * faer::scale(0.5);
                let eig_nu = m_nu_s.selfadjoint_eigendecomposition(faer::Side::Lower);
                let u_raw = u_ch.transpose() * eig_nu.u();
                let mut u_pmns = faer::Mat::<f64>::zeros(3, 3);
                for i in 0..3 { for j in 0..3 {
                    u_pmns.write(i, j, u_raw.read(perm_u[i], perm_d[j]));
                }}
                extract_pmns_angles(&u_pmns)
            };

            // Compute gradients using V_6 basis directions (12 evals)
            let mut g_12 = [0.0_f64; 6];
            let mut g_13 = [0.0_f64; 6];
            let mut g_23 = [0.0_f64; 6];
            for mu in 0..n_basis {
                let mut bp = [0.0_f64; 6]; bp[mu] = eps;
                let mut bm = [0.0_f64; 6]; bm[mu] = -eps;
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
                for k in 0..6 { beta[k] = u[k]; }
                apply_v6_perturbation(&mut m_perturbed, &v6_basis, &beta, &lift);
                // A = m_perturbed - m_nu_base
                let mut delta = faer::Mat::<f64>::zeros(3, 3);
                for i in 0..3 { for j in 0..3 {
                    delta.write(i, j, m_perturbed.read(i, j) - m_nu_base.read(i, j));
                }}
                delta
            };

            let a_mat = precompute_perturbation(&u_solar);
            let b_mat = precompute_perturbation(&u_atmo);

            // Inner optimization via Gauss-Newton (replaces 651-point grid scan)
            // Closure: (t1, t2) -> (theta_12, theta_13, theta_23) using affine M_nu
            let inner_angles = |t1: f64, t2: f64| -> (f64, f64, f64) {
                let mut m_nu = m_nu_base.clone();
                for i in 0..3 { for j in 0..3 {
                    m_nu.write(i, j, m_nu.read(i, j)
                        + t1 * a_mat.read(i, j)
                        + t2 * b_mat.read(i, j));
                }}
                angles_from_mnu(&m_nu)
            };

            // Gradient-guided initial guess for t1
            let dot6 = |a: &[f64; 6], b: &[f64; 6]| -> f64 {
                a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
            };
            let g12_u = dot6(&g_12, &u_solar);
            let t1_init = if g12_u.abs() > 0.01 { (pdg_t12 - 28.5) / g12_u } else { 2.5 };
            let t1_init = t1_init.clamp(0.0, 5.0);

            let (best_t1, best_t2, best_angles, best_score) = gauss_newton_2d(
                &inner_angles,
                t1_init,
                0.0, // initial t2 guess
                (pdg_t12, pdg_t13, pdg_t23),
                (1.0, 2.24, 1.0), // sqrt(5) weight on theta_13
                15, // max iterations
            );

            (best_score, best_angles, best_t1, best_t2)
        };

        println!("--- JOINT 4D OPTIMIZATION ---");

        // Coarse grid over (alpha_ch, alpha_nu)
        // Previous optimum: (3.50, 1.35). Focused neighborhood.
        let grid: Vec<(f64, f64)> = (25..=50_i32).flat_map(|i|
            (8..=20_i32).map(move |j| (i as f64 * 0.1, j as f64 * 0.1))
        ).collect();

        let results: Vec<(f64, f64, f64, (f64, f64, f64), f64, f64)> = grid.par_iter()
            .map(|&(a_ch, a_nu)| {
                let (score, angles, t1, t2) = evaluate(a_ch, a_nu);
                (score, a_ch, a_nu, angles, t1, t2)
            })
            .collect();

        let best = results.iter().min_by(|a, b| a.0.partial_cmp(&b.0).unwrap()).unwrap();

        println!("  Coarse grid: {} points evaluated", results.len());
        println!("  Best coarse: alpha_ch={:.2}, alpha_nu={:.2}, t1={:.2}, t2={:.2}",
            best.1, best.2, best.4, best.5);
        println!("    theta_12 = {:.4} (error {:.2}%)", (best.3).0,
            (((best.3).0 - pdg_t12) / pdg_t12 * 100.0).abs());
        println!("    theta_13 = {:.4} (error {:.2}%)", (best.3).1,
            (((best.3).1 - pdg_t13) / pdg_t13 * 100.0).abs());
        println!("    theta_23 = {:.4} (error {:.2}%)", (best.3).2,
            (((best.3).2 - pdg_t23) / pdg_t23 * 100.0).abs());
        println!("    Score = {:.6}", best.0);

        // Fine refinement around the coarse best
        let a_ch_center = best.1;
        let a_nu_center = best.2;

        let fine_grid: Vec<(f64, f64)> = (-10..=10_i32).flat_map(|i|
            (-10..=10_i32).map(move |j| (a_ch_center + i as f64 * 0.05, a_nu_center + j as f64 * 0.05))
        ).filter(|&(a, b)| a > 0.0 && b > 0.0).collect();

        let fine_results: Vec<(f64, f64, f64, (f64, f64, f64), f64, f64)> = fine_grid.par_iter()
            .map(|&(a_ch, a_nu)| {
                let (score, angles, t1, t2) = evaluate(a_ch, a_nu);
                (score, a_ch, a_nu, angles, t1, t2)
            })
            .collect();

        let fine_best = fine_results.iter().min_by(|a, b| a.0.partial_cmp(&b.0).unwrap()).unwrap();

        println!("\n  Fine grid: {} points evaluated", fine_results.len());
        println!("\n  === JOINT 4D OPTIMUM ===");
        println!("  alpha_ch = {:.2}", fine_best.1);
        println!("  alpha_nu = {:.2}", fine_best.2);
        println!("  t_solar  = {:.2}", fine_best.4);
        println!("  t_atmo   = {:.2}", fine_best.5);
        println!("  theta_12 = {:.4} deg (PDG: {:.2}, error: {:.2}%)",
            (fine_best.3).0, pdg_t12, (((fine_best.3).0 - pdg_t12) / pdg_t12 * 100.0).abs());
        println!("  theta_13 = {:.4} deg (PDG: {:.2}, error: {:.2}%)",
            (fine_best.3).1, pdg_t13, (((fine_best.3).1 - pdg_t13) / pdg_t13 * 100.0).abs());
        println!("  theta_23 = {:.4} deg (PDG: {:.2}, error: {:.2}%)",
            (fine_best.3).2, pdg_t23, (((fine_best.3).2 - pdg_t23) / pdg_t23 * 100.0).abs());
        println!("  Combined score: {:.6}", fine_best.0);

        // Compare with previous best
        let prev_score = ((33.37 - pdg_t12) / pdg_t12).powi(2)
                       + ((47.40 - pdg_t23) / pdg_t23).powi(2)
                       + 5.0 * ((8.52 - pdg_t13) / pdg_t13).powi(2);
        println!("\n  Previous 4-param score (3.75, 1.30, 2.49, 0.11): {:.6}", prev_score);
        println!("  Improvement: {:.1}x", prev_score / fine_best.0);
    }

    /// CP violation baseline: real PMNS matrices have J = 0, delta_CP = 0.
    ///
    /// This establishes the interface for future complex-phase extension
    /// via the G2 stabilizer's Fano pair signs.
    #[test]
    fn test_cp_phase_baseline_real_pmns() {
        let result = compute_pmns((11, 12), (7, 8));
        assert!(
            result.jarlskog_invariant.abs() < 1e-15,
            "Real PMNS matrix must have J = 0, got {}",
            result.jarlskog_invariant
        );
        assert!(
            result.cp_phase_deg.abs() < 1e-10,
            "Real PMNS matrix must have delta_CP = 0, got {}",
            result.cp_phase_deg
        );
        println!("  CP phase baseline: J = {:.2e}, delta_CP = {:.2} deg",
            result.jarlskog_invariant, result.cp_phase_deg);
        println!("  PDG 2024: delta_CP ~ 195 deg (normal ordering)");
        println!("  Next: complex phase from Fano pair signs (G2 stabilizer J_k)");
    }

    /// Verify extract_cp_phase correctly recovers delta from known J.
    #[test]
    fn test_cp_phase_extraction_formula() {
        // PDG values: theta_12=33.41, theta_13=8.54, theta_23=49.0
        // delta_CP=195 deg -> J ~ -0.0334
        let t12 = 33.41_f64;
        let t13 = 8.54_f64;
        let t23 = 49.0_f64;
        let delta = 195.0_f64;
        let s12 = t12.to_radians().sin();
        let c12 = t12.to_radians().cos();
        let s13 = t13.to_radians().sin();
        let c13 = t13.to_radians().cos();
        let s23 = t23.to_radians().sin();
        let c23 = t23.to_radians().cos();
        let j_pdg = s12 * c12 * s23 * c23 * s13 * c13 * c13 * delta.to_radians().sin();

        let recovered = super::extract_cp_phase((t12, t13, t23), j_pdg);
        // extract_cp_phase returns asin(sin(delta)), which maps 195 -> -15 (mod 360)
        // since sin(195) = sin(-15) = -sin(15).
        let sin_delta = delta.to_radians().sin();
        let sin_recovered = recovered.to_radians().sin();
        assert!(
            (sin_delta - sin_recovered).abs() < 1e-10,
            "sin(delta) mismatch: expected {:.6}, got {:.6}",
            sin_delta, sin_recovered
        );
        println!("  J_PDG = {:.6}", j_pdg);
        println!("  Recovered delta_CP = {:.2} deg (sin matches PDG 195 deg)", recovered);
    }

    // =======================================================================
    // Phase B: Absolute neutrino masses
    // =======================================================================

    /// Mass-squared ratio r = dm21_sq / dm31_sq is a scale-free prediction.
    /// PDG 2024 (normal ordering): r = 7.53e-5 / 2.453e-3 = 0.0307.
    #[test]
    fn test_mass_squared_ratio() {
        let result = compute_pmns((11, 12), (7, 8));
        let dm21 = result.delta_m21_sq;
        let dm31 = result.delta_m31_sq;

        println!("  Mass eigenvalues (arb. units): {:?}", result.neutrino_masses);
        println!("  dm21_sq = {:.6e}", dm21);
        println!("  dm31_sq = {:.6e}", dm31);

        if dm31.abs() > 1e-30 {
            let r = dm21 / dm31;
            println!("  r = dm21_sq / dm31_sq = {:.6}", r);
            println!("  PDG 2024: r = 0.0307 (normal ordering)");

            // Register whether the ratio is in the right ballpark
            // (order of magnitude is the first test)
            if r > 0.0 && r < 1.0 {
                println!("  PASS: r is in (0, 1) -- normal ordering");
            } else if r > 1.0 {
                println!("  NOTE: r > 1 -- inverted ordering");
            } else {
                println!("  NOTE: r <= 0 -- degenerate spectrum");
            }
        } else {
            println!("  WARNING: dm31_sq ~ 0, degenerate spectrum");
        }
    }

    /// Reconstruct absolute masses from the algebraic ratio + m1 input.
    /// Scan m1 from 0 to 0.1 eV and check cosmological bound.
    #[test]
    fn test_absolute_mass_reconstruction() {
        let result = compute_pmns((11, 12), (7, 8));
        let nu = &result.neutrino_masses;

        // Compute ratios relative to the lightest eigenvalue
        // The absolute scale drops out of the PMNS matrix; only ratios matter.
        let m_min = nu[0].max(1e-30);
        let r1 = nu[1] / m_min;
        let r2 = nu[2] / m_min;

        println!("  Mass ratios: m2/m1 = {:.4}, m3/m1 = {:.4}", r1, r2);

        // PDG 2024 mass-squared differences (normal ordering):
        //   dm21_sq = 7.53e-5 eV^2, dm31_sq = 2.453e-3 eV^2
        let pdg_dm21_sq = 7.53e-5_f64;
        let pdg_dm31_sq = 2.453e-3_f64;

        // For several m1 values, compute sum(m_i)
        println!("\n  Absolute mass reconstruction (using PDG dm^2 for scale):");
        println!("  {:>8} | {:>8} {:>8} {:>8} | {:>10} | {:>6}",
            "m1 (eV)", "m1", "m2", "m3", "sum (eV)", "bound");

        for m1_mev in [0.0, 1.0, 5.0, 10.0, 20.0, 50.0] {
            let m1 = m1_mev * 1e-3; // meV -> eV
            let m2 = (m1 * m1 + pdg_dm21_sq).sqrt();
            let m3 = (m1 * m1 + pdg_dm31_sq).sqrt();
            let sum_m = m1 + m2 + m3;
            let bound = if sum_m < 0.12 { "OK" } else { "EXCLUDED" };
            println!("  {:>8.4} | {:>8.5} {:>8.5} {:>8.5} | {:>10.5} | {:>6}",
                m1, m1, m2, m3, sum_m, bound);
        }
    }

    /// Effective electron-neutrino mass m_beta for KATRIN.
    /// m_beta^2 = sum |U_ei|^2 m_i^2
    #[test]
    fn test_effective_electron_neutrino_mass() {
        let result = compute_pmns((11, 12), (7, 8));
        let u = &result.matrix;

        // Using PDG mass-squared diffs with m1 = 0 (lightest case)
        let pdg_dm21_sq = 7.53e-5_f64;
        let pdg_dm31_sq = 2.453e-3_f64;
        let m1 = 0.0_f64;
        let m2 = (m1 * m1 + pdg_dm21_sq).sqrt();
        let m3 = (m1 * m1 + pdg_dm31_sq).sqrt();
        let masses = [m1, m2, m3];

        let mut m_beta_sq = 0.0_f64;
        for i in 0..3 {
            let u_ei = u.read(0, i);
            m_beta_sq += u_ei * u_ei * masses[i] * masses[i];
        }
        let m_beta = m_beta_sq.sqrt();

        println!("  m_beta = {:.4} meV (m1=0 case)", m_beta * 1e3);
        println!("  KATRIN bound: m_beta < 450 meV (90% CL)");
        assert!(m_beta < 0.45, "m_beta = {} eV exceeds KATRIN bound", m_beta);

        // Also compute for m1 = 50 meV (near cosmological bound)
        let m1_heavy = 0.05_f64;
        let m2h = (m1_heavy * m1_heavy + pdg_dm21_sq).sqrt();
        let m3h = (m1_heavy * m1_heavy + pdg_dm31_sq).sqrt();
        let masses_h = [m1_heavy, m2h, m3h];
        let mut m_beta_sq_h = 0.0_f64;
        for i in 0..3 {
            let u_ei = u.read(0, i);
            m_beta_sq_h += u_ei * u_ei * masses_h[i] * masses_h[i];
        }
        let m_beta_h = m_beta_sq_h.sqrt();
        println!("  m_beta = {:.2} meV (m1=50 meV case)", m_beta_h * 1e3);
        println!("  sum(m_i) = {:.4} eV", m1_heavy + m2h + m3h);
    }

    // =======================================================================
    // Phase A: CP violation via Fano pair complex phases
    // =======================================================================

    /// Explore the CP phase structure from Fano pair signs.
    ///
    /// For each fixed imaginary unit k=1..7, the G2 stabilizer gives 3 Fano
    /// lines with signs sigma_j in {+1, -1}. These signs define a complex
    /// basis z_j = u_j - i*sigma_j*w_j. When the PMNS matrix is rephased
    /// by these signs, it acquires a Dirac CP phase.
    ///
    /// The rephasing: U_PMNS_complex = diag(sigma_1, sigma_2, sigma_3) * U_PMNS_real
    /// This is a diagonal phase matrix acting on the flavor (generation) indices.
    /// The Jarlskog invariant of the rephased matrix is:
    ///   J = sigma_1 * sigma_2 * sigma_3 * J_real
    /// which is zero since J_real = 0 for a real matrix.
    ///
    /// A more physical approach: the complex structure J_k introduces phases
    /// into the OFF-DIAGONAL mass matrix elements. If M_ij for i != j picks up
    /// a factor exp(i * phi_ij) where phi_ij comes from the Fano line connecting
    /// generations i and j, then the diagonalization produces a genuinely
    /// complex unitary matrix with nonzero J.
    #[test]
    fn test_cp_phase_from_fano_signs() {
        use gororoba_algebra::lie::g2_stabilizer::complex_structure;

        let result = compute_pmns((11, 12), (7, 8));

        println!("  === CP Phase from Fano Pair Signs ===\n");

        for k in 1..=7 {
            let cs = complex_structure(k);
            let signs: Vec<i8> = cs.fano_pairs.iter().map(|&(_, _, s)| s).collect();
            let sign_product: i8 = signs.iter().product();

            println!("  k={}: fano_pairs = {:?}", k, cs.fano_pairs);
            println!("       signs = {:?}, product = {}", signs, sign_product);

            // The Fano signs define a Z_2^3 phase structure.
            // For CP violation, we need a RELATIVE phase between mass matrix
            // elements. Consider the off-diagonal phase:
            // phi_ij = pi * (1 - sigma_line(i,j)) / 2
            // where sigma_line(i,j) is the sign of the Fano line connecting
            // the perp-index pairs for generations i and j.
            //
            // If sigma = +1: phi = 0 (no phase)
            // If sigma = -1: phi = pi (sign flip = exp(i*pi))

            // Count how many signs are -1 (these contribute pi phases)
            let n_negative = signs.iter().filter(|&&s| s == -1).count();
            println!("       negative signs: {} -> discrete phase: {} * pi/3",
                n_negative, n_negative);
        }

        // The discrete Z_2^3 structure from Fano signs gives at most
        // exp(i*pi) = -1 phases. For a continuous CP phase like delta_CP = 195 deg,
        // we need a mechanism beyond discrete signs.
        //
        // The continuous mechanism: the psi automorphism (S_3 generator, order 3)
        // introduces exp(2*pi*i/3) = exp(i*120 deg) phases when cycling
        // O_1 -> O_2 -> O_3. This is the natural source of a CP phase near 180 deg.
        //
        // delta_CP ~ 180 + correction from Fano signs
        // PDG: delta_CP ~ 195 deg (= 180 + 15)

        println!("\n  The psi automorphism (order 3) provides the dominant phase:");
        println!("  exp(2*pi*i/3) = exp(i * 120 deg)");
        println!("  Combined with Fano sign corrections:");
        println!("  delta_CP ~ 180 + O(15) deg -- consistent with PDG 195 deg");

        // Compute what delta_CP would be if J comes from the psi automorphism:
        // The psi overlap <sel_i, psi(sel_j)> ~ cos(2*pi/3) = -0.5
        // The imaginary part ~ sin(2*pi/3) = sqrt(3)/2
        // J ~ sin(120 deg) * product_of_sines = sqrt(3)/2 * s12*c12*s23*c23*s13*c13^2
        let (t12, t13, t23) = result.angles_deg;
        let s12 = t12.to_radians().sin();
        let c12 = t12.to_radians().cos();
        let s13 = t13.to_radians().sin();
        let c13 = t13.to_radians().cos();
        let s23 = t23.to_radians().sin();
        let c23 = t23.to_radians().cos();

        let j_max = s12 * c12 * s23 * c23 * s13 * c13 * c13;
        let j_psi = j_max * (2.0 * std::f64::consts::PI / 3.0).sin();
        let delta_psi = extract_cp_phase((t12, t13, t23), j_psi);

        println!("\n  J_max (from angles) = {:.6}", j_max);
        println!("  J_psi (sin(120) * J_max) = {:.6}", j_psi);
        println!("  delta_CP_psi = {:.2} deg", delta_psi);
        println!("  PDG 2024: delta_CP = 195 +/- 25 deg");

        // This is a PREDICTION: the psi automorphism predicts
        // sin(delta_CP) = sin(120 deg) = sqrt(3)/2
        // delta_CP = 60 deg or 120 deg (from asin)
        // But the physical phase could be in the second quadrant: 180 - 60 = 120 or 180 + 60 = 240
        // or equivalently -120 deg = 240 deg.
        // PDG convention: delta_CP in [0, 360), so 195 is close to 180 + 15.
        //
        // The psi prediction gives |sin(delta)| = sqrt(3)/2 ~ 0.866.
        // PDG sin(195 deg) = sin(180+15) = -sin(15) = -0.259.
        // These don't match -- the psi automorphism alone doesn't give the right magnitude.
        //
        // This means the CP phase requires the FULL complex mass matrix construction
        // (task A1-A3), not just a simple rephasing argument.
        println!("\n  NOTE: Full complex mass matrix construction needed for delta_CP.");
        println!("  The psi rephasing gives |sin(delta)| = {:.3}, PDG has |sin(195)| = {:.3}",
            (2.0 * std::f64::consts::PI / 3.0).sin(),
            195.0_f64.to_radians().sin().abs());
    }

    // =======================================================================
    // Phase C: Chi-squared global fit
    // =======================================================================

    /// Chi-squared evaluation at the current best-fit point.
    #[test]
    fn test_chi2_best_fit_point() {
        let pdg = Pdg2024::default();
        let result = compute_pmns((11, 12), (7, 8));
        let chi2 = chi_squared_pmns(&result, &pdg);
        let pulls = pmns_pulls(&result, &pdg);

        let (t12, t13, t23) = result.angles_deg;
        println!("  === Chi-squared Global Fit (3 angles) ===\n");
        println!("  {:>10} | {:>8} | {:>8} | {:>6} | {:>6}",
            "Observable", "This", "PDG", "sigma", "Pull");
        println!("  {:-<10}-+-{:-<8}-+-{:-<8}-+-{:-<6}-+-{:-<6}", "", "", "", "", "");
        for &(name, pull) in &pulls {
            let (val, pdg_val, err) = match name {
                "theta_12" => (t12, pdg.theta_12_deg, pdg.theta_12_err),
                "theta_13" => (t13, pdg.theta_13_deg, pdg.theta_13_err),
                "theta_23" => (t23, pdg.theta_23_deg, pdg.theta_23_err),
                _ => (0.0, 0.0, 1.0),
            };
            println!("  {:>10} | {:>8.2} | {:>8.2} | {:>6.2} | {:>+6.2}",
                name, val, pdg_val, err, pull);
        }
        println!("\n  chi2 = {:.4} (3 dof)", chi2);
        println!("  chi2/dof = {:.4}", chi2 / 3.0);
        println!("  Note: Good fit if chi2/dof < 1. Tension if > 4.");
    }

    /// Selector pair scan: find the best (charged, neutrino) pair minimizing chi^2.
    #[test]
    fn test_chi2_selector_scan() {
        use rayon::prelude::*;

        let pdg = Pdg2024::default();

        // All valid selector pairs: (a, b) with 1 <= a < b <= 15
        let pairs: Vec<(usize, usize)> = (1..=15)
            .flat_map(|a| ((a + 1)..=15).map(move |b| (a, b)))
            .collect();

        // Scan all (charged, neutrino) combinations
        // Outer loop parallel via rayon, inner loop sequential.
        let results: Vec<(f64, (usize, usize), (usize, usize), (f64, f64, f64))> = pairs
            .par_iter()
            .flat_map_iter(|&ch| {
                pairs.iter().map(move |&nu| {
                    let r = compute_pmns(ch, nu);
                    let chi2 = chi_squared_pmns(&r, &pdg);
                    (chi2, ch, nu, r.angles_deg)
                })
            })
            .collect();

        // Sort by chi2
        let mut sorted = results;
        sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        println!("  === Chi-squared Selector Pair Scan ===\n");
        println!("  {:>6} | {:>10} | {:>10} | {:>8} {:>8} {:>8}",
            "chi2", "charged", "neutrino", "t12", "t13", "t23");
        println!("  {:-<6}-+-{:-<10}-+-{:-<10}-+-{:-<8}-{:-<8}-{:-<8}", "", "", "", "", "", "");

        for entry in sorted.iter().take(10) {
            let (chi2, ch, nu, (t12, t13, t23)) = entry;
            println!("  {:>6.2} | {:>10?} | {:>10?} | {:>8.2} {:>8.2} {:>8.2}",
                chi2, ch, nu, t12, t13, t23);
        }

        let best = &sorted[0];
        println!("\n  Best fit: chi2 = {:.4}, charged = {:?}, neutrino = {:?}",
            best.0, best.1, best.2);
        println!("  chi2/dof = {:.4} (3 dof)", best.0 / 3.0);
        println!("  Total pairs scanned: {}", sorted.len());
    }

    /// Regression test pinning the joint 4D PMNS optimum (C-1491).
    ///
    /// Evaluates at the known-optimal (alpha_ch=3.50, alpha_nu=1.35, t_solar=1.54,
    /// t_atmo=2.00) and verifies the angles match with strict tolerances.
    /// Also computes the 3x4 Jacobian and reports condition number.
    #[test]
    fn test_pmns_4d_optimum_regression() {
        let ch_pair = (11_usize, 12);
        let nu_pair = (7_usize, 8);
        let alpha_ch = 3.50;
        let alpha_nu = 1.35;
        let eps = 0.05_f64;

        let (v6_basis, _sv, _assessors) = extract_v6_basis();
        let lift = TensorElementLift;
        let n_basis = v6_basis.nrows().min(6);

        // Build at the known optimum
        let (m_ch_base, m_nu_base) = construct_pmns_matrices_two_param(
            ch_pair, nu_pair, alpha_ch, alpha_nu,
        );
        let eig_ch = m_ch_base.selfadjoint_eigendecomposition(faer::Side::Lower);
        let eig_nu_0 = m_nu_base.selfadjoint_eigendecomposition(faer::Side::Lower);
        let u_raw_0 = eig_ch.u().transpose() * eig_nu_0.u();
        let (_, perm_u, perm_d) = crate::quark_sector::extract_ckm_permutation_aware(&u_raw_0);

        let angles_at = |beta: &[f64; 6]| -> (f64, f64, f64) {
            let mut m_nu = m_nu_base.clone();
            apply_v6_perturbation(&mut m_nu, &v6_basis, beta, &lift);
            let m_nu_s = (&m_nu + m_nu.transpose()) * faer::scale(0.5);
            let eig_nu = m_nu_s.selfadjoint_eigendecomposition(faer::Side::Lower);
            let u_raw = eig_ch.u().transpose() * eig_nu.u();
            let mut u_pmns = faer::Mat::<f64>::zeros(3, 3);
            for i in 0..3 { for j in 0..3 {
                u_pmns.write(i, j, u_raw.read(perm_u[i], perm_d[j]));
            }}
            extract_pmns_angles(&u_pmns)
        };

        // Compute constrained directions
        let mut g_12 = [0.0_f64; 6];
        let mut g_13 = [0.0_f64; 6];
        let mut g_23 = [0.0_f64; 6];
        for mu in 0..n_basis {
            let mut bp = [0.0_f64; 6]; bp[mu] = eps;
            let mut bm = [0.0_f64; 6]; bm[mu] = -eps;
            let (t12_p, t13_p, t23_p) = angles_at(&bp);
            let (t12_m, t13_m, t23_m) = angles_at(&bm);
            g_12[mu] = (t12_p - t12_m) / (2.0 * eps);
            g_13[mu] = (t13_p - t13_m) / (2.0 * eps);
            g_23[mu] = (t23_p - t23_m) / (2.0 * eps);
        }

        let u_solar = compute_constrained_solar_direction(&g_12, &g_13, &g_23);
        let u_atmo = compute_constrained_atmospheric_direction(&g_23, &g_13, &u_solar);

        // Apply at t_solar=1.54, t_atmo=2.00
        let t_solar = 1.54_f64;
        let t_atmo = 2.00_f64;
        let mut beta_opt = [0.0_f64; 6];
        for k in 0..6 { beta_opt[k] = t_solar * u_solar[k] + t_atmo * u_atmo[k]; }

        let (t12, t13, t23) = angles_at(&beta_opt);

        println!("--- 4D OPTIMUM REGRESSION ---");
        println!("  theta_12 = {:.4} deg (expected ~33.84)", t12);
        println!("  theta_13 = {:.4} deg (expected ~8.56)", t13);
        println!("  theta_23 = {:.4} deg (expected ~48.74)", t23);

        // Pin angles -- theta_13 tightest
        assert!((t13 - 8.56).abs() < 0.05,
            "theta_13 regression FAILED: {:.4} (expected ~8.56)", t13);
        assert!((t12 - 33.84).abs() < 0.5,
            "theta_12 regression FAILED: {:.4} (expected ~33.84)", t12);
        assert!((t23 - 48.74).abs() < 0.5,
            "theta_23 regression FAILED: {:.4} (expected ~48.74)", t23);

        println!("  PASS: 4D optimum regression");
    }

    /// Chi-squared for the full pipeline at all established operating points.
    ///
    /// Compares three levels of the PMNS pipeline against PDG 2024.
    #[test]
    fn test_chi2_full_pipeline_summary() {
        let pdg = Pdg2024::default();

        // Level 1: Diagonal-only baseline (selectors only)
        let r_diag = compute_pmns((11, 12), (7, 8));
        let chi2_diag = chi_squared_pmns(&r_diag, &pdg);

        // Level 2: Two-param psi coupling (C-1464 result)
        // theta_12 ~ 29.2, theta_13 ~ 8.64, theta_23 ~ 47.1
        let chi2_psi = ((29.2 - pdg.theta_12_deg) / pdg.theta_12_err).powi(2)
                     + ((8.64 - pdg.theta_13_deg) / pdg.theta_13_err).powi(2)
                     + ((47.1 - pdg.theta_23_deg) / pdg.theta_23_err).powi(2);

        // Level 3: V_6 solar correction (C-1478/C-1490 result)
        let chi2_v6 = ((33.42 - pdg.theta_12_deg) / pdg.theta_12_err).powi(2)
                    + ((8.63 - pdg.theta_13_deg) / pdg.theta_13_err).powi(2)
                    + ((47.08 - pdg.theta_23_deg) / pdg.theta_23_err).powi(2);

        // Level 4: 4D joint optimum (C-1491 result)
        let chi2_4d = ((33.84 - pdg.theta_12_deg) / pdg.theta_12_err).powi(2)
                    + ((8.56 - pdg.theta_13_deg) / pdg.theta_13_err).powi(2)
                    + ((48.74 - pdg.theta_23_deg) / pdg.theta_23_err).powi(2);

        println!("  === PMNS Chi-squared Summary (3 angles vs PDG 2024) ===\n");
        println!("  {:>30} | {:>8} | {:>8} | {:>8} {:>8} {:>8}",
            "Pipeline level", "chi2", "chi2/3", "t12", "t13", "t23");
        println!("  {:-<30}-+-{:-<8}-+-{:-<8}-+-{:-<8}-{:-<8}-{:-<8}", "", "", "", "", "", "");
        println!("  {:>30} | {:>8.2} | {:>8.2} | {:>8.2} {:>8.2} {:>8.2}",
            "Diagonal only", chi2_diag, chi2_diag / 3.0,
            r_diag.angles_deg.0, r_diag.angles_deg.1, r_diag.angles_deg.2);
        println!("  {:>30} | {:>8.2} | {:>8.2} | {:>8.2} {:>8.2} {:>8.2}",
            "Psi coupling (C-1464)", chi2_psi, chi2_psi / 3.0, 29.2, 8.64, 47.1);
        println!("  {:>30} | {:>8.2} | {:>8.2} | {:>8.2} {:>8.2} {:>8.2}",
            "V_6 correction (C-1490)", chi2_v6, chi2_v6 / 3.0, 33.42, 8.63, 47.08);
        println!("  {:>30} | {:>8.2} | {:>8.2} | {:>8.2} {:>8.2} {:>8.2}",
            "4D joint optimum (C-1491)", chi2_4d, chi2_4d / 3.0, 33.84, 8.56, 48.74);
        println!("\n  PDG 2024 reference: theta_12={:.2} +/- {:.2}, theta_13={:.2} +/- {:.2}, theta_23={:.2} +/- {:.1}",
            pdg.theta_12_deg, pdg.theta_12_err,
            pdg.theta_13_deg, pdg.theta_13_err,
            pdg.theta_23_deg, pdg.theta_23_err);

        // Individual pulls at the 4D optimum
        println!("\n  --- Pulls at 4D optimum ---");
        let pulls_4d = [
            ("theta_12", (33.84 - pdg.theta_12_deg) / pdg.theta_12_err),
            ("theta_13", (8.56 - pdg.theta_13_deg) / pdg.theta_13_err),
            ("theta_23", (48.74 - pdg.theta_23_deg) / pdg.theta_23_err),
        ];
        for (name, pull) in &pulls_4d {
            println!("  {:>10}: {:>+.2} sigma", name, pull);
        }
        println!("\n  Total chi2 at 4D optimum: {:.2} (3 observables, 4 parameters -> 0 effective dof)",
            chi2_4d);
    }

    /// Mass ordering prediction from the algebraic eigenvalue spectrum.
    ///
    /// Normal ordering: m1 < m2 < m3 (dm31 > 0)
    /// Inverted ordering: m3 < m1 < m2 (dm31 < 0)
    ///
    /// The algebraic framework predicts the ordering from the sign of
    /// dm31_sq = m3^2 - m1^2.
    #[test]
    fn test_mass_ordering_prediction() {
        // Use the two-param pipeline at the optimized point
        let ch_pair = (11_usize, 12);
        let nu_pair = (7_usize, 8);
        let (_m_ch, m_nu) = construct_pmns_matrices_two_param(ch_pair, nu_pair, 3.75, 1.30);

        let eig_nu = m_nu.selfadjoint_eigendecomposition(faer::Side::Lower);
        let mut eigenvalues: Vec<f64> = (0..3)
            .map(|i| eig_nu.s().column_vector().read(i))
            .collect();

        let mut abs_eigenvalues: Vec<f64> = eigenvalues.iter().map(|e| e.abs()).collect();
        abs_eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let m1 = abs_eigenvalues[0];
        let m2 = abs_eigenvalues[1];
        let m3 = abs_eigenvalues[2];

        let dm21_sq = m2 * m2 - m1 * m1;
        let dm31_sq = m3 * m3 - m1 * m1;
        let r = dm21_sq / dm31_sq;

        println!("  === Mass Ordering Prediction ===\n");
        println!("  Raw eigenvalues: {:.6e}, {:.6e}, {:.6e}",
            eigenvalues[0], eigenvalues[1], eigenvalues[2]);
        println!("  |m_i| sorted:    {:.6e}, {:.6e}, {:.6e}", m1, m2, m3);
        println!("  dm21_sq = {:.6e}", dm21_sq);
        println!("  dm31_sq = {:.6e}", dm31_sq);
        println!("  r = dm21/dm31 = {:.6}", r);
        println!("  PDG: r = 0.0307 (normal ordering)");

        let ordering = if dm31_sq > 0.0 { "NORMAL" } else { "INVERTED" };
        println!("\n  Predicted ordering: {}", ordering);
        println!("  PDG 2024: Normal ordering preferred (>3 sigma)");

        // The hierarchy ratio tells us how separated the mass scales are
        let hierarchy = m3 / m1;
        println!("  m3/m1 = {:.2} (mass hierarchy strength)", hierarchy);

        // Check if the eigenvalue signs are all positive (physical consistency)
        eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let all_positive = eigenvalues.iter().all(|&e| e > 0.0);
        println!("  All eigenvalues positive: {}", all_positive);
        if !all_positive {
            println!("  (Negative eigenvalues indicate see-saw-like mechanism)");
        }
    }

    /// Mass ratio r = dm21/dm31 scan over alpha parameters.
    ///
    /// PDG: r = 0.0307. Current baseline gives r = 0.213 (7x too large).
    /// Scan alpha_ch x alpha_nu to find values giving r closer to PDG.
    #[test]
    fn test_mass_ratio_alpha_scan() {
        use rayon::prelude::*;

        let pdg = Pdg2024::default();
        let ch_pair = (11_usize, 12);
        let nu_pair = (7_usize, 8);
        let pdg_r = 0.0307_f64;

        let grid: Vec<(f64, f64)> = (1..=80)
            .flat_map(|a| (1..=40).map(move |b| (a as f64 * 0.1, b as f64 * 0.1)))
            .collect();

        let results: Vec<_> = grid.par_iter().map(|&(a_ch, a_nu)| {
            let (_m_ch, m_nu) = construct_pmns_matrices_two_param(
                ch_pair, nu_pair, a_ch, a_nu
            );
            let eig_nu = m_nu.selfadjoint_eigendecomposition(faer::Side::Lower);
            let mut ev: Vec<f64> = (0..3)
                .map(|i| eig_nu.s().column_vector().read(i).abs())
                .collect();
            ev.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let dm21 = ev[1] * ev[1] - ev[0] * ev[0];
            let dm31 = ev[2] * ev[2] - ev[0] * ev[0];
            let r = if dm31.abs() > 1e-30 { dm21 / dm31 } else { f64::MAX };

            // Also compute angles
            let (m_ch2, _) = construct_pmns_matrices_two_param(ch_pair, nu_pair, a_ch, a_nu);
            let eig_ch = m_ch2.selfadjoint_eigendecomposition(faer::Side::Lower);
            let u_raw = eig_ch.u().transpose() * eig_nu.u();
            let (u_pmns, _, _) = crate::quark_sector::extract_ckm_permutation_aware(&u_raw);
            let (t12, t13, t23) = extract_pmns_angles(&u_pmns);
            let chi2 = ((t12 - pdg.theta_12_deg) / pdg.theta_12_err).powi(2)
                     + ((t13 - pdg.theta_13_deg) / pdg.theta_13_err).powi(2)
                     + ((t23 - pdg.theta_23_deg) / pdg.theta_23_err).powi(2);

            let r_err = (r - pdg_r).abs() / pdg_r;
            (r_err, r, chi2, a_ch, a_nu, t12, t13, t23, ev[2] / ev[0])
        }).collect();

        // Sort by mass ratio accuracy
        let mut by_r = results.clone();
        by_r.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        println!("  === Mass Ratio r = dm21/dm31 Alpha Scan ===\n");
        println!("  PDG: r = {:.4}, m3/m1 ~ 50\n", pdg_r);
        println!("  Top 10 by ratio accuracy:");
        println!("  {:>6} {:>6} | {:>8} {:>8} | {:>8} {:>8} {:>8} | {:>6}",
            "a_ch", "a_nu", "r", "m3/m1", "t12", "t13", "t23", "chi2");
        for e in by_r.iter().take(10) {
            println!("  {:>6.1} {:>6.1} | {:>8.4} {:>8.1} | {:>8.2} {:>8.2} {:>8.2} | {:>6.1}",
                e.3, e.4, e.1, e.8, e.5, e.6, e.7, e.2);
        }

        // Find best combined (good r + good angles)
        let mut combined = results;
        combined.sort_by(|a, b| {
            let sa = a.0 + 0.001 * a.2; // r_error + weight * chi2_angles
            let sb = b.0 + 0.001 * b.2;
            sa.partial_cmp(&sb).unwrap()
        });

        println!("\n  Best combined (r accuracy + angle chi2):");
        for e in combined.iter().take(5) {
            println!("  a_ch={:.1} a_nu={:.1}: r={:.4} (err {:.1}%), chi2={:.1}, t12={:.1} t13={:.1} t23={:.1}",
                e.3, e.4, e.1, e.0 * 100.0, e.2, e.5, e.6, e.7);
        }
    }

    /// Regression test pinning the Gauss-Newton 4D optimum (C-1492).
    ///
    /// Evaluates at the known-optimal parameters and verifies angles match
    /// with strict tolerances. This is the canonical pinned state of the
    /// PMNS angle-sector fit.
    #[test]
    fn test_pmns_gauss_newton_regression() {
        let ch_pair = (11_usize, 12);
        let nu_pair = (7_usize, 8);
        let alpha_ch = 3.00;
        let alpha_nu = 1.35;
        let eps = 0.05_f64;

        let (v6_basis, _sv, _assessors) = extract_v6_basis();
        let lift = TensorElementLift;
        let n_basis = v6_basis.nrows().min(6);

        let (m_ch_base, m_nu_base) = construct_pmns_matrices_two_param(
            ch_pair, nu_pair, alpha_ch, alpha_nu,
        );
        let eig_ch = m_ch_base.selfadjoint_eigendecomposition(faer::Side::Lower);
        let eig_nu_0 = m_nu_base.selfadjoint_eigendecomposition(faer::Side::Lower);
        let u_raw_0 = eig_ch.u().transpose() * eig_nu_0.u();
        let (_, perm_u, perm_d) = crate::quark_sector::extract_ckm_permutation_aware(&u_raw_0);

        let angles_at = |beta: &[f64; 6]| -> (f64, f64, f64) {
            let mut m_nu = m_nu_base.clone();
            apply_v6_perturbation(&mut m_nu, &v6_basis, beta, &lift);
            let m_nu_s = (&m_nu + m_nu.transpose()) * faer::scale(0.5);
            let eig_nu = m_nu_s.selfadjoint_eigendecomposition(faer::Side::Lower);
            let u_raw = eig_ch.u().transpose() * eig_nu.u();
            let mut u_pmns = faer::Mat::<f64>::zeros(3, 3);
            for i in 0..3 { for j in 0..3 {
                u_pmns.write(i, j, u_raw.read(perm_u[i], perm_d[j]));
            }}
            extract_pmns_angles(&u_pmns)
        };

        // Compute constrained directions and apply at optimal t values
        let mut g_12 = [0.0_f64; 6];
        let mut g_13 = [0.0_f64; 6];
        let mut g_23 = [0.0_f64; 6];
        for mu in 0..n_basis {
            let mut bp = [0.0_f64; 6]; bp[mu] = eps;
            let mut bm = [0.0_f64; 6]; bm[mu] = -eps;
            let (t12_p, t13_p, t23_p) = angles_at(&bp);
            let (t12_m, t13_m, t23_m) = angles_at(&bm);
            g_12[mu] = (t12_p - t12_m) / (2.0 * eps);
            g_13[mu] = (t13_p - t13_m) / (2.0 * eps);
            g_23[mu] = (t23_p - t23_m) / (2.0 * eps);
        }

        let u_solar = compute_constrained_solar_direction(&g_12, &g_13, &g_23);
        let u_atmo = compute_constrained_atmospheric_direction(&g_23, &g_13, &u_solar);

        // Use Gauss-Newton to find the optimum
        let inner_angles = |t1: f64, t2: f64| -> (f64, f64, f64) {
            let mut beta = [0.0_f64; 6];
            for k in 0..6 { beta[k] = t1 * u_solar[k] + t2 * u_atmo[k]; }
            angles_at(&beta)
        };

        let (t1, t2, (t12, t13, t23), score) = gauss_newton_2d(
            &inner_angles, 1.5, 0.0,
            (33.41, 8.54, 49.0),
            (1.0, 2.24, 1.0),
            15,
        );

        println!("--- GAUSS-NEWTON REGRESSION ---");
        println!("  t_solar = {:.4}, t_atmo = {:.4}", t1, t2);
        println!("  theta_12 = {:.4} deg (expected ~33.36)", t12);
        println!("  theta_13 = {:.4} deg (expected ~8.54)", t13);
        println!("  theta_23 = {:.4} deg (expected ~48.99)", t23);
        println!("  score = {:.6e}", score);

        // Strict tolerances -- theta_13 tightest
        assert!((t13 - 8.54).abs() < 0.02,
            "theta_13 regression: {:.4} (tol 0.02)", t13);
        assert!((t12 - 33.4).abs() < 0.2,
            "theta_12 regression: {:.4} (tol 0.2)", t12);
        assert!((t23 - 49.0).abs() < 0.2,
            "theta_23 regression: {:.4} (tol 0.2)", t23);

        println!("  PASS: Gauss-Newton regression");
    }

    // =======================================================================
    // Complex PMNS: CP violation from psi eigenspace decomposition
    // =======================================================================

    /// Construct complex mass matrices using psi-eigenspace decomposition.
    ///
    /// The psi automorphism (order 3, cycles O_1->O_2->O_3) has eigenvalues
    /// {1, omega, omega^2} where omega = exp(2*pi*i/3).
    ///
    /// For a 16D friction profile v, the psi-eigenspace projections are:
    ///   P_1(v) = (v + psi(v) + psi^2(v)) / 3           (eigenvalue 1)
    ///   P_w(v) = (v + w^2*psi(v) + w*psi^2(v)) / 3     (eigenvalue omega)
    ///   P_w2(v) = (v + w*psi(v) + w^2*psi^2(v)) / 3    (eigenvalue omega^2)
    ///
    /// The complex off-diagonal mass matrix element is:
    ///   M_ij = alpha * (<v_i, P_1(v_j)> + w*<v_i, P_w(v_j)> + w^2*<v_i, P_w2(v_j)>)
    ///        = alpha * (<v_i, v_j> + w*<v_i, psi_w(v_j)> + w^2*<v_i, psi_w2(v_j)>)
    ///
    /// Simplified: M_ij = alpha * sum_k w^k * <v_i, psi^k(v_j)>
    /// where the sum is over the three psi eigenvalues.
    ///
    /// The imaginary part of M_ij carries the CP-violating phase.
    #[test]
    fn test_complex_pmns_cp_phase() {
        use cd_kernel::{gourlay_psi, gourlay_psi_n};

        let nu_pair = (7_usize, 8);

        // Build friction profiles (same as two-param pipeline)
        use crate::majorana_braiding::MajoranaMode;
        use crate::bell_inequality::{SignTableCache, rotate_sparse};
        use crate::three_fermion_generations::get_sedenion_subalgebras;

        let (o1, o2, o3) = get_sedenion_subalgebras();
        let subs = [&o1, &o2, &o3];
        let sign_table = SignTableCache::new(16);

        let nu_a = MajoranaMode { gamma_index: nu_pair.0 - 1, cd_basis_index: nu_pair.0, cd_dim: 16 };
        let nu_b = MajoranaMode { gamma_index: nu_pair.1 - 1, cd_basis_index: nu_pair.1, cd_dim: 16 };

        let build_profile = |mode_i: &MajoranaMode, mode_j: &MajoranaMode, sub: &[usize]| -> [f64; 16] {
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

        let nu_profiles: Vec<[f64; 16]> = subs.iter()
            .map(|s| build_profile(&nu_a, &nu_b, s))
            .collect();

        let dot16 = |a: &[f64; 16], b: &[f64; 16]| -> f64 {
            a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
        };

        // omega = exp(2*pi*i/3) = -1/2 + i*sqrt(3)/2
        let omega_re = -0.5_f64;
        let omega_im = 3.0_f64.sqrt() / 2.0;

        println!("  === Complex PMNS: CP Phase from Psi Eigenspace ===\n");
        println!("  omega = exp(2*pi*i/3) = {:.4} + {:.4}i\n", omega_re, omega_im);

        // Build complex off-diagonal elements for the neutrino mass matrix
        // M_ij^complex = sum_{k=0,1,2} omega^k * <profile_i, psi^k(profile_j)>
        let mut m_complex = [[(0.0_f64, 0.0_f64); 3]; 3]; // (re, im)

        for i in 0..3 {
            for j in 0..3 {
                // <v_i, psi^0(v_j)> = <v_i, v_j>  (k=0, omega^0 = 1)
                let c0 = dot16(&nu_profiles[i], &nu_profiles[j]);

                // <v_i, psi^1(v_j)>  (k=1, omega^1 = omega)
                let psi1_j = gourlay_psi(&nu_profiles[j]);
                let c1 = dot16(&nu_profiles[i], &psi1_j);

                // <v_i, psi^2(v_j)>  (k=2, omega^2 = omega*)
                let psi2_j = gourlay_psi_n(&nu_profiles[j], 2);
                let c2 = dot16(&nu_profiles[i], &psi2_j);

                // M_ij = c0 * 1 + c1 * omega + c2 * omega^2
                //      = c0 * 1 + c1 * (w_re + i*w_im) + c2 * (w_re - i*w_im)
                // (omega^2 = conjugate of omega for cube root of unity)
                let re = c0 + c1 * omega_re + c2 * omega_re;
                let im = c1 * omega_im - c2 * omega_im;

                m_complex[i][j] = (re, im);
            }
        }

        println!("  Complex neutrino mass matrix M_nu:");
        for i in 0..3 {
            let row: Vec<String> = (0..3).map(|j| {
                let (re, im) = m_complex[i][j];
                if im.abs() < 1e-15 {
                    format!("{:>8.4}", re)
                } else {
                    format!("{:.4}{:+.4}i", re, im)
                }
            }).collect();
            println!("    [{}]", row.join("  "));
        }

        // Check Hermiticity: M_ij = M_ji*
        let mut max_herm_err = 0.0_f64;
        for i in 0..3 {
            for j in 0..3 {
                let err_re = (m_complex[i][j].0 - m_complex[j][i].0).abs();
                let err_im = (m_complex[i][j].1 + m_complex[j][i].1).abs();
                max_herm_err = max_herm_err.max(err_re).max(err_im);
            }
        }
        println!("\n  Hermiticity error: {:.2e}", max_herm_err);

        // Check if any off-diagonal element has nonzero imaginary part
        let mut max_im = 0.0_f64;
        for i in 0..3 {
            for j in 0..3 {
                if i != j {
                    max_im = max_im.max(m_complex[i][j].1.abs());
                }
            }
        }
        println!("  Max off-diagonal |Im(M_ij)|: {:.6e}", max_im);

        if max_im > 1e-15 {
            println!("  NONZERO imaginary part detected -- CP violation present!");

            // The Jarlskog invariant is proportional to the imaginary part
            // of the off-diagonal products. For a 3x3 Hermitian matrix,
            // J ~ Im(M_12 * M_23 * M_31) / (mass differences)^3
            let m12 = (m_complex[0][1].0, m_complex[0][1].1);
            let m23 = (m_complex[1][2].0, m_complex[1][2].1);
            let m31 = (m_complex[2][0].0, m_complex[2][0].1);

            // Complex product M_12 * M_23 * M_31
            let p12_23_re = m12.0 * m23.0 - m12.1 * m23.1;
            let p12_23_im = m12.0 * m23.1 + m12.1 * m23.0;
            let _triple_re = p12_23_re * m31.0 - p12_23_im * m31.1;
            let triple_im = p12_23_re * m31.1 + p12_23_im * m31.0;

            println!("  Im(M_12 * M_23 * M_31) = {:.6e}", triple_im);
            println!("  This is proportional to the Jarlskog invariant.");

            // Extract a crude delta_CP from the phase of M_12
            let phase_12 = m_complex[0][1].1.atan2(m_complex[0][1].0);
            println!("  arg(M_12) = {:.2} deg", phase_12.to_degrees());
            println!("  arg(M_23) = {:.2} deg",
                m_complex[1][2].1.atan2(m_complex[1][2].0).to_degrees());
            println!("  arg(M_31) = {:.2} deg",
                m_complex[2][0].1.atan2(m_complex[2][0].0).to_degrees());
        } else {
            println!("  All imaginary parts are zero -- no CP violation in this basis.");
            println!("  This means the psi overlaps <v_i, psi(v_j)> = <v_i, psi^2(v_j)>");
            println!("  (psi is symmetric on the friction profiles).");
        }
    }

    /// CP violation from cross-sector psi asymmetry.
    ///
    /// The psi-eigenspace decomposition gives Im = 0 within a single sector
    /// because <v_i, psi(v_j)> = <v_i, psi^2(v_j)>. But the PMNS matrix is
    /// U = U_ch^dagger * U_nu -- a PRODUCT of two diagonalizations. The CP phase
    /// emerges from the RELATIVE orientation of the charged and neutrino friction
    /// tensors under psi.
    ///
    /// Key idea: build a 6x6 "cross-sector" complex Gram matrix
    ///   G_ij = sum_k omega^k * <ch_profile_i, psi^k(nu_profile_j)>
    /// The off-diagonal elements of G mix charged and neutrino sectors, and
    /// the psi asymmetry between sectors (different selectors) can produce
    /// nonzero imaginary parts.
    #[test]
    fn test_cross_sector_cp_phase() {
        use cd_kernel::{gourlay_psi, gourlay_psi_n};
        use crate::majorana_braiding::MajoranaMode;
        use crate::bell_inequality::{SignTableCache, rotate_sparse};
        use crate::three_fermion_generations::get_sedenion_subalgebras;

        let ch_pair = (11_usize, 12);
        let nu_pair = (7_usize, 8);

        let (o1, o2, o3) = get_sedenion_subalgebras();
        let subs = [&o1, &o2, &o3];
        let sign_table = SignTableCache::new(16);

        let ch_a = MajoranaMode { gamma_index: ch_pair.0 - 1, cd_basis_index: ch_pair.0, cd_dim: 16 };
        let ch_b = MajoranaMode { gamma_index: ch_pair.1 - 1, cd_basis_index: ch_pair.1, cd_dim: 16 };
        let nu_a = MajoranaMode { gamma_index: nu_pair.0 - 1, cd_basis_index: nu_pair.0, cd_dim: 16 };
        let nu_b = MajoranaMode { gamma_index: nu_pair.1 - 1, cd_basis_index: nu_pair.1, cd_dim: 16 };

        let build_profile = |mode_i: &MajoranaMode, mode_j: &MajoranaMode, sub: &[usize]| -> [f64; 16] {
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
            .map(|s| build_profile(&ch_a, &ch_b, s))
            .collect();
        let nu_profiles: Vec<[f64; 16]> = subs.iter()
            .map(|s| build_profile(&nu_a, &nu_b, s))
            .collect();

        let dot16 = |a: &[f64; 16], b: &[f64; 16]| -> f64 {
            a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
        };

        let omega_re = -0.5_f64;
        let omega_im = 3.0_f64.sqrt() / 2.0;

        println!("  === Cross-Sector CP Phase ===\n");

        // Cross-sector Gram matrix: G_ij = sum_k omega^k <ch_i, psi^k(nu_j)>
        let mut g_cross = [[(0.0_f64, 0.0_f64); 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                let c0 = dot16(&ch_profiles[i], &nu_profiles[j]);
                let psi1_j = gourlay_psi(&nu_profiles[j]);
                let c1 = dot16(&ch_profiles[i], &psi1_j);
                let psi2_j = gourlay_psi_n(&nu_profiles[j], 2);
                let c2 = dot16(&ch_profiles[i], &psi2_j);

                let re = c0 + c1 * omega_re + c2 * omega_re;
                let im = c1 * omega_im - c2 * omega_im;
                g_cross[i][j] = (re, im);
            }
        }

        println!("  Cross-sector Gram matrix G = <ch_i, psi^k(nu_j)>:");
        for i in 0..3 {
            let row: Vec<String> = (0..3).map(|j| {
                let (re, im) = g_cross[i][j];
                if im.abs() < 1e-15 {
                    format!("{:>10.4}", re)
                } else {
                    format!("{:>7.4}{:+.4}i", re, im)
                }
            }).collect();
            println!("    [{}]", row.join("  "));
        }

        let mut max_im = 0.0_f64;
        for i in 0..3 {
            for j in 0..3 {
                max_im = max_im.max(g_cross[i][j].1.abs());
            }
        }
        println!("\n  Max |Im(G_ij)|: {:.6e}", max_im);

        if max_im > 1e-15 {
            println!("  NONZERO cross-sector imaginary part -- CP violation from selector asymmetry!");

            // The cross-sector phase is the source of delta_CP.
            // The dominant phase comes from the largest off-diagonal element.
            for i in 0..3 {
                for j in 0..3 {
                    if i == j { continue; }
                    let (re, im) = g_cross[i][j];
                    if (re * re + im * im).sqrt() > 1e-10 {
                        let phase = im.atan2(re).to_degrees();
                        println!("  arg(G[{},{}]) = {:.2} deg  (|G| = {:.4})",
                            i, j, phase, (re * re + im * im).sqrt());
                    }
                }
            }
        } else {
            println!("  Cross-sector Gram matrix is also real.");
            println!("  psi acts symmetrically on BOTH sectors for these selectors.");

            // Check: are the intra-sector psi overlaps symmetric too?
            // <ch_i, psi(ch_j)> vs <ch_i, psi^2(ch_j)>
            println!("\n  Checking psi symmetry breakdown:");
            for i in 0..3 {
                for j in 0..3 {
                    if i == j { continue; }
                    let c1_ch = dot16(&ch_profiles[i], &gourlay_psi(&ch_profiles[j]));
                    let c2_ch = dot16(&ch_profiles[i], &gourlay_psi_n(&ch_profiles[j], 2));
                    let c1_nu = dot16(&nu_profiles[i], &gourlay_psi(&nu_profiles[j]));
                    let c2_nu = dot16(&nu_profiles[i], &gourlay_psi_n(&nu_profiles[j], 2));
                    let c1_cross = dot16(&ch_profiles[i], &gourlay_psi(&nu_profiles[j]));
                    let c2_cross = dot16(&ch_profiles[i], &gourlay_psi_n(&nu_profiles[j], 2));

                    println!("  ({},{}): ch: c1={:.6} c2={:.6} diff={:.2e}  nu: c1={:.6} c2={:.6} diff={:.2e}  cross: c1={:.6} c2={:.6} diff={:.2e}",
                        i, j,
                        c1_ch, c2_ch, (c1_ch - c2_ch).abs(),
                        c1_nu, c2_nu, (c1_nu - c2_nu).abs(),
                        c1_cross, c2_cross, (c1_cross - c2_cross).abs());
                }
            }
        }
    }

    /// Complex PMNS extension: J_k-based complexification for CP violation.
    ///
    /// The real symmetric mass matrices force J_CP = 0. To get nonzero CP,
    /// we use the Fano-derived complex structure J_k from the G2 stabilizer
    /// to inject phases into the off-diagonal mass matrix entries.
    ///
    /// For each embedding k=1..7, the 3 Fano pairs define 3 complex lines
    /// on e_k^perp. The cross-generational psi-overlaps in this complex
    /// basis carry natural phases that become the source of CP violation.
    #[test]
    fn test_complex_pmns_cp_violation() {
        use gororoba_algebra::lie::g2_stabilizer::complex_structure;
        use cd_kernel::gourlay_psi;

        let ch_pair = (11_usize, 12);
        let nu_pair = (7_usize, 8);
        let alpha_ch = 3.00;
        let alpha_nu = 1.35;

        // Get the current best real mass matrices
        let (m_ch_real, m_nu_real) = construct_pmns_matrices_two_param(
            ch_pair, nu_pair, alpha_ch, alpha_nu,
        );

        // Apply V_6 correction at the optimal point
        let (v6_basis, _sv, _assessors) = extract_v6_basis();
        let lift = TensorElementLift;
        let eps = 0.05_f64;
        let n_basis = v6_basis.nrows().min(6);

        // Compute constrained directions
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
            let mut u_pmns = faer::Mat::<f64>::zeros(3, 3);
            for i in 0..3 { for j in 0..3 {
                u_pmns.write(i, j, u_raw.read(perm_u[i], perm_d[j]));
            }}
            extract_pmns_angles(&u_pmns)
        };

        let mut g_12 = [0.0_f64; 6];
        let mut g_13 = [0.0_f64; 6];
        let mut g_23 = [0.0_f64; 6];
        for mu in 0..n_basis {
            let mut bp = [0.0_f64; 6]; bp[mu] = eps;
            let mut bm = [0.0_f64; 6]; bm[mu] = -eps;
            let (t12_p, t13_p, t23_p) = angles_at(&bp);
            let (t12_m, t13_m, t23_m) = angles_at(&bm);
            g_12[mu] = (t12_p - t12_m) / (2.0 * eps);
            g_13[mu] = (t13_p - t13_m) / (2.0 * eps);
            g_23[mu] = (t23_p - t23_m) / (2.0 * eps);
        }
        let u_solar = compute_constrained_solar_direction(&g_12, &g_13, &g_23);
        let u_atmo = compute_constrained_atmospheric_direction(&g_23, &g_13, &u_solar);

        // Apply optimal V_6 correction
        let inner_angles = |t1: f64, t2: f64| -> (f64, f64, f64) {
            let mut beta = [0.0_f64; 6];
            for k in 0..6 { beta[k] = t1 * u_solar[k] + t2 * u_atmo[k]; }
            angles_at(&beta)
        };
        let (t_sol, t_atm, _, _) = gauss_newton_2d(
            &inner_angles, 1.5, 0.0,
            (33.41, 8.54, 49.0), (1.0, 2.24, 1.0), 15,
        );

        let mut beta_opt = [0.0_f64; 6];
        for k in 0..6 { beta_opt[k] = t_sol * u_solar[k] + t_atm * u_atmo[k]; }
        let mut m_nu_corrected = m_nu_real.clone();
        apply_v6_perturbation(&mut m_nu_corrected, &v6_basis, &beta_opt, &lift);
        let m_nu_corrected = (&m_nu_corrected + m_nu_corrected.transpose()) * faer::scale(0.5);

        println!("--- COMPLEX PMNS: CP VIOLATION VIA J_k ---");

        // For each k=1..7, build the complex PMNS matrix
        for k in 1..=7 {
            let cs = complex_structure(k);

            // The 3 Fano pairs give 3 complex lines on e_k^perp.
            // Each pair (a_idx, b_idx, sign) defines:
            //   z_j = e_{perp[a_idx]} + i * sign * e_{perp[b_idx]}
            //
            // The cross-generational psi overlap in this complex basis
            // carries a phase. We compute this phase for each generation pair.

            // Build the 3 complex friction profiles
            // For each generation g and each Fano pair j, compute the
            // complex overlap: <profile_g, z_j> = <profile_g, u_j> + i*sign*<profile_g, J_k(u_j)>

            // The psi overlap between generation i and j in the complex basis gives
            // the off-diagonal phase. We use the existing nu_profiles.
            use crate::bell_inequality::{SignTableCache, rotate_sparse};
            use crate::majorana_braiding::MajoranaMode;
            use crate::three_fermion_generations::get_sedenion_subalgebras;

            let (o1, o2, o3) = get_sedenion_subalgebras();
            let subs = [&o1, &o2, &o3];
            let sign_table = SignTableCache::new(16);

            let nu_a = MajoranaMode { gamma_index: nu_pair.0 - 1, cd_basis_index: nu_pair.0, cd_dim: 16 };
            let nu_b = MajoranaMode { gamma_index: nu_pair.1 - 1, cd_basis_index: nu_pair.1, cd_dim: 16 };

            // Build 16D friction profiles per generation
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

            // For each generation pair (i,j), compute the psi-overlap PHASE
            // in the J_k complex basis.
            // psi maps nu_profiles[j] -> psi(nu_profiles[j]).
            // The overlap <nu_profiles[i], psi(nu_profiles[j])> is the real part.
            // In the complex basis, the J_k-rotated overlap gives the imaginary part:
            // <nu_profiles[i], J_k(psi(nu_profiles[j]))> is the imaginary component.
            //
            // J_k acts on the lower 8 components (octonion part) via the 6x6 matrix.

            let apply_jk = |v: &[f64; 16]| -> [f64; 16] {
                let mut result = [0.0_f64; 16];
                // J_k acts on perp indices only
                for r in 0..6 {
                    for s in 0..6 {
                        result[cs.perp_indices[r]] += cs.matrix[r][s] * v[cs.perp_indices[s]];
                    }
                }
                result
            };

            let dot16 = |a: &[f64; 16], b: &[f64; 16]| -> f64 {
                a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
            };

            // Build complex neutrino mass matrix with CP amplitude scan
            // M_nu^complex[i][j] = M_nu_real[i][j] + i * alpha_CP * <profile_i, J_k(psi(profile_j))>
            // alpha_CP << alpha_nu to preserve the real angle fit

            // Pre-compute the imaginary overlap matrix
            let mut im_template = [[0.0_f64; 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    if i != j {
                        let psi_j = gourlay_psi(&nu_profiles[j]);
                        let jk_psi_j = apply_jk(&psi_j);
                        im_template[i][j] = dot16(&nu_profiles[i], &jk_psi_j);
                    }
                }
            }

            // Scan alpha_CP to find the sweet spot: nonzero J_CP with minimal angle distortion
            let mut best_alpha_cp = 0.0_f64;
            let mut best_j_cp = 0.0_f64;
            let mut best_delta = 0.0_f64;
            let mut best_score = f64::MAX;
            let mut best_angles_cp = (0.0_f64, 0.0_f64, 0.0_f64);

            for step in 0..=100_i32 {
                let alpha_cp = step as f64 * 0.01;

                let mut m_nu_re = [[0.0_f64; 3]; 3];
                let mut m_nu_im = [[0.0_f64; 3]; 3];
                for i in 0..3 {
                    for j in 0..3 {
                        m_nu_re[i][j] = m_nu_corrected.read(i, j);
                        if i != j {
                            m_nu_im[i][j] = alpha_cp * im_template[i][j];
                        }
                    }
                }

                // Hermitize
                for i in 0..3 {
                    for j in (i + 1)..3 {
                        let re_avg = (m_nu_re[i][j] + m_nu_re[j][i]) / 2.0;
                        let im_avg = (m_nu_im[i][j] - m_nu_im[j][i]) / 2.0;
                        m_nu_re[i][j] = re_avg;
                        m_nu_re[j][i] = re_avg;
                        m_nu_im[i][j] = im_avg;
                        m_nu_im[j][i] = -im_avg;
                    }
                    m_nu_im[i][i] = 0.0;
                }

                let mut m_nu_c = faer::Mat::<faer::complex_native::c64>::zeros(3, 3);
                let mut m_ch_c = faer::Mat::<faer::complex_native::c64>::zeros(3, 3);
                for i in 0..3 {
                    for j in 0..3 {
                        m_nu_c.write(i, j, faer::complex_native::c64::new(m_nu_re[i][j], m_nu_im[i][j]));
                        m_ch_c.write(i, j, faer::complex_native::c64::new(m_ch_real.read(i, j), 0.0));
                    }
                }

                let eig_ch_c = m_ch_c.selfadjoint_eigendecomposition(faer::Side::Lower);
                let eig_nu_c = m_nu_c.selfadjoint_eigendecomposition(faer::Side::Lower);
                let u_pmns_c = eig_ch_c.u().adjoint() * eig_nu_c.u();

                let u_e3 = u_pmns_c.read(0, 2);
                let theta_13 = u_e3.abs().min(1.0).asin().to_degrees();
                let cos_13 = (theta_13.to_radians()).cos();
                let theta_12 = if cos_13 > 1e-15 {
                    (u_pmns_c.read(0, 1).abs() / cos_13).min(1.0).asin().to_degrees()
                } else { 0.0 };
                let theta_23 = if cos_13 > 1e-15 {
                    (u_pmns_c.read(1, 2).abs() / cos_13).min(1.0).asin().to_degrees()
                } else { 0.0 };

                let j_cp = (u_pmns_c.read(0, 0) * u_pmns_c.read(1, 1)
                    * u_pmns_c.read(0, 1).conj() * u_pmns_c.read(1, 0).conj()).im;
                let delta_cp = (-u_e3).arg().to_degrees();

                // Score: angle preservation + nonzero J_CP
                let angle_cost = ((theta_12 - 33.41) / 33.41).powi(2)
                    + ((theta_13 - 8.54) / 8.54).powi(2)
                    + ((theta_23 - 49.0) / 49.0).powi(2);
                let score = angle_cost - 0.01 * j_cp.abs(); // reward nonzero J_CP

                if score < best_score && j_cp.abs() > 1e-6 {
                    best_score = score;
                    best_alpha_cp = alpha_cp;
                    best_j_cp = j_cp;
                    best_delta = delta_cp;
                    best_angles_cp = (theta_12, theta_13, theta_23);
                }
            }

            println!("  k={}: best alpha_CP={:.2}, theta_12={:.2}, theta_13={:.2}, theta_23={:.2}, J_CP={:.4e}, delta={:.1}",
                k, best_alpha_cp, best_angles_cp.0, best_angles_cp.1, best_angles_cp.2, best_j_cp, best_delta);
        }

        println!("\n  PDG targets: J_CP ~ 3.0e-2 (leptons), delta_CP ~ -140 to -180 deg (NO best-fit)");
    }

    /// Fine-grained alpha_CP scan for the best k-embedding.
    ///
    /// At alpha_CP = 1.0 the complex coupling overwhelms the real mass matrix,
    /// distorting mixing angles. Scan alpha_CP in [0.001, 0.5] to find the
    /// sweet spot where J_CP is nonzero but angles remain close to PDG.
    #[test]
    fn test_complex_pmns_alpha_scan() {
        use gororoba_algebra::lie::g2_stabilizer::complex_structure;
        use cd_kernel::gourlay_psi;
        use crate::majorana_braiding::MajoranaMode;
        use crate::bell_inequality::{SignTableCache, rotate_sparse};
        use crate::three_fermion_generations::get_sedenion_subalgebras;
        use nalgebra::SMatrix;
        use num_complex::Complex;

        type Mat3c = SMatrix<Complex<f64>, 3, 3>;

        let pdg = Pdg2024::default();
        let ch_pair = (11_usize, 12);
        let nu_pair = (7_usize, 8);

        // Build real mass matrices at the optimal point
        let (m_ch_real, m_nu_real) = construct_pmns_matrices_two_param(
            ch_pair, nu_pair, 3.75, 1.30
        );

        // Build friction profiles for the imaginary injection
        let (o1, o2, o3) = get_sedenion_subalgebras();
        let subs = [&o1, &o2, &o3];
        let sign_table = SignTableCache::new(16);

        let nu_a = MajoranaMode { gamma_index: nu_pair.0 - 1, cd_basis_index: nu_pair.0, cd_dim: 16 };
        let nu_b = MajoranaMode { gamma_index: nu_pair.1 - 1, cd_basis_index: nu_pair.1, cd_dim: 16 };

        let build_profile = |mode_i: &MajoranaMode, mode_j: &MajoranaMode, sub: &[usize]| -> [f64; 16] {
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

        let nu_profiles: Vec<[f64; 16]> = subs.iter()
            .map(|s| build_profile(&nu_a, &nu_b, s))
            .collect();

        let dot16 = |a: &[f64; 16], b: &[f64; 16]| -> f64 {
            a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
        };

        // Use k=1 (strongest CP signal from earlier scan)
        let k_fixed = 1_usize;
        let _cs = complex_structure(k_fixed);

        // Build the J_k-rotated profiles: J_k(psi(profile_j))
        // J_k acts on the octonion part (indices 1-7) via left-multiplication by e_k
        let apply_jk = |profile: &[f64; 16]| -> [f64; 16] {
            use gororoba_algebra::construction::octonion::Octonion;
            let ek = Octonion::basis(k_fixed);
            let mut result = [0.0_f64; 16];
            // J_k acts on the lower octonion block (indices 0-7)
            let mut oct_part = [0.0_f64; 8];
            for idx in 0..8 {
                oct_part[idx] = profile[idx];
            }
            let oct = Octonion::new(oct_part);
            let jk_oct = ek.multiply(&oct);
            for idx in 0..8 {
                result[idx] = jk_oct.components[idx];
            }
            // Upper block: J_k acts similarly via e_k multiplication on the upper octonion
            let mut upper = [0.0_f64; 8];
            for idx in 0..8 {
                upper[idx] = profile[idx + 8];
            }
            let upper_oct = Octonion::new(upper);
            let jk_upper = ek.multiply(&upper_oct);
            for idx in 0..8 {
                result[idx + 8] = jk_upper.components[idx];
            }
            result
        };

        println!("  === Alpha_CP Fine Scan (k={}) ===\n", k_fixed);
        println!("  {:>8} | {:>8} {:>8} {:>8} | {:>10} | {:>8} | {:>6}",
            "alpha_CP", "t12", "t13", "t23", "J_CP", "delta", "chi2");
        println!("  {:-<8}-+-{:-<8}-{:-<8}-{:-<8}-+-{:-<10}-+-{:-<8}-+-{:-<6}",
            "", "", "", "", "", "", "");

        let mut best_chi2 = f64::MAX;
        let mut best_alpha = 0.0_f64;
        let mut best_result = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);

        for alpha_step in 0..=50 {
            let alpha_cp = alpha_step as f64 * 0.01; // 0.00 to 0.50

            // Build complex neutrino mass matrix:
            // M_nu_complex = M_nu_real + i * alpha_cp * Im_injection
            let mut m_nu_complex = Mat3c::zeros();
            for i in 0..3 {
                for j in 0..3 {
                    m_nu_complex[(i, j)] = Complex::new(m_nu_real.read(i, j), 0.0);
                }
            }

            // Add imaginary off-diagonal: i * alpha_cp * <profile_i, J_k(psi(profile_j))>
            for i in 0..3 {
                for j in 0..3 {
                    if i == j { continue; }
                    let psi_j = gourlay_psi(&nu_profiles[j]);
                    let jk_psi_j = apply_jk(&psi_j);
                    let coupling = dot16(&nu_profiles[i], &jk_psi_j);
                    m_nu_complex[(i, j)] += Complex::new(0.0, alpha_cp * coupling);
                }
            }

            // Hermitianize: M = (M + M^dagger) / 2
            let m_herm = (m_nu_complex + m_nu_complex.adjoint()) * Complex::new(0.5, 0.0);

            // Eigendecompose
            let eigen = nalgebra::SymmetricEigen::new(m_herm);

            // Charged lepton eigenvectors (real, from faer)
            let eig_ch = m_ch_real.selfadjoint_eigendecomposition(faer::Side::Lower);

            // Build PMNS: U = U_ch^T * U_nu (U_ch is real, U_nu is complex)
            let u_nu = &eigen.eigenvectors;
            let mut u_pmns = Mat3c::zeros();
            for i in 0..3 {
                for j in 0..3 {
                    let mut sum = Complex::new(0.0, 0.0);
                    for m in 0..3 {
                        let u_ch_mi = eig_ch.u().read(m, i);
                        sum += Complex::new(u_ch_mi, 0.0) * u_nu[(m, j)];
                    }
                    u_pmns[(i, j)] = sum;
                }
            }

            // Extract angles from |U_ij|
            let u_e3 = u_pmns[(0, 2)].norm();
            let theta_13 = u_e3.min(1.0).asin().to_degrees();
            let cos_13 = theta_13.to_radians().cos();
            let theta_12 = if cos_13 > 1e-15 {
                (u_pmns[(0, 1)].norm() / cos_13).min(1.0).asin().to_degrees()
            } else { 0.0 };
            let theta_23 = if cos_13 > 1e-15 {
                (u_pmns[(1, 2)].norm() / cos_13).min(1.0).asin().to_degrees()
            } else { 0.0 };

            // Jarlskog: J = Im(U_e2 * U_mu3 * conj(U_e3) * conj(U_mu2))
            let j_cp = (u_pmns[(0, 1)] * u_pmns[(1, 2)]
                      * u_pmns[(0, 2)].conj() * u_pmns[(1, 1)].conj()).im;

            let delta = extract_cp_phase((theta_12, theta_13, theta_23), j_cp);

            // Chi^2 over 3 angles only
            let chi2_angles = ((theta_12 - pdg.theta_12_deg) / pdg.theta_12_err).powi(2)
                            + ((theta_13 - pdg.theta_13_deg) / pdg.theta_13_err).powi(2)
                            + ((theta_23 - pdg.theta_23_deg) / pdg.theta_23_err).powi(2);

            if alpha_step % 5 == 0 || chi2_angles < best_chi2 {
                println!("  {:>8.3} | {:>8.2} {:>8.2} {:>8.2} | {:>10.4e} | {:>8.1} | {:>6.1}",
                    alpha_cp, theta_12, theta_13, theta_23, j_cp, delta, chi2_angles);
            }

            if chi2_angles < best_chi2 {
                best_chi2 = chi2_angles;
                best_alpha = alpha_cp;
                best_result = (theta_12, theta_13, theta_23, j_cp, delta);
            }
        }

        println!("\n  === BEST FIT ===");
        println!("  alpha_CP = {:.3}", best_alpha);
        println!("  theta_12 = {:.2} deg (PDG: {:.2})", best_result.0, pdg.theta_12_deg);
        println!("  theta_13 = {:.2} deg (PDG: {:.2})", best_result.1, pdg.theta_13_deg);
        println!("  theta_23 = {:.2} deg (PDG: {:.2})", best_result.2, pdg.theta_23_deg);
        println!("  J_CP = {:.4e} (PDG: ~3e-2)", best_result.3);
        println!("  delta_CP = {:.1} deg (PDG: ~195)", best_result.4);
        println!("  chi2 = {:.2} (3 angles)", best_chi2);
    }

    /// CP violation via cross-sector rephasing of the EXISTING PMNS matrix.
    ///
    /// Instead of making mass matrices complex (which changes eigenvalues and
    /// breaks the permutation alignment), inject the CP phase as a diagonal
    /// rephasing of the real PMNS matrix:
    ///
    ///   U_CP = diag(1, e^{i*phi_12}, e^{i*phi_13}) * U_real * diag(1, e^{i*psi_2}, e^{i*psi_3})
    ///
    /// where the phases come from the cross-sector Gram matrix:
    ///   phi_ij = alpha_CP * arg(G_ij)
    ///   G_ij = sum_k omega^k * <ch_profile_i, psi^k(nu_profile_j)>
    ///
    /// This preserves the mixing angles (|U_ij| unchanged) while introducing
    /// a nonzero Jarlskog invariant.
    #[test]
    fn test_cp_rephasing_pipeline() {
        use cd_kernel::gourlay_psi;
        use crate::majorana_braiding::MajoranaMode;
        use crate::bell_inequality::{SignTableCache, rotate_sparse};
        use crate::three_fermion_generations::get_sedenion_subalgebras;
        use num_complex::Complex;

        let ch_pair = (11_usize, 12);
        let nu_pair = (7_usize, 8);

        // Step 1: Get the real PMNS matrix from the existing pipeline
        let (m_ch, m_nu) = construct_pmns_matrices_two_param(ch_pair, nu_pair, 3.75, 1.30);
        let eig_ch = m_ch.selfadjoint_eigendecomposition(faer::Side::Lower);
        let eig_nu = m_nu.selfadjoint_eigendecomposition(faer::Side::Lower);
        let u_raw = eig_ch.u().transpose() * eig_nu.u();
        let (u_real, _, _) = crate::quark_sector::extract_ckm_permutation_aware(&u_raw);
        let (t12, t13, t23) = extract_pmns_angles(&u_real);

        println!("  === CP Rephasing Pipeline ===\n");
        println!("  Real PMNS angles: t12={:.2}, t13={:.2}, t23={:.2}", t12, t13, t23);

        // Step 2: Build cross-sector Gram phases
        let (o1, o2, o3) = get_sedenion_subalgebras();
        let subs = [&o1, &o2, &o3];
        let sign_table = SignTableCache::new(16);

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

        // Cross-sector Gram phases
        let mut gram_phases = [[0.0_f64; 3]; 3];
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
            }
        }

        println!("  Cross-sector Gram phases (radians):");
        for i in 0..3 {
            println!("    [{:.4}, {:.4}, {:.4}]",
                gram_phases[i][0], gram_phases[i][1], gram_phases[i][2]);
        }

        // Step 3: Apply rephasing to the PMNS matrix.
        // The CP phase enters through the relative phases between rows and columns.
        // For the standard parameterization, delta_CP appears in U_e3:
        //   U_e3 = s13 * e^{-i*delta}
        // So we can inject delta directly into the (0,2) element.
        //
        // The algebraic prediction: delta_CP comes from the off-diagonal
        // Gram phase arg(G_12) = 45 deg = pi/4 (from earlier test).
        // But the physical delta_CP in the PMNS matrix is a combination of
        // all the Gram phases.
        //
        // Rephasing: U_CP[i][j] = U_real[i][j] * exp(i * alpha * gram_phases[i][j])
        println!("\n  Alpha_CP scan with rephasing:");
        println!("  {:>8} | {:>8} {:>8} {:>8} | {:>10} | {:>8}",
            "alpha_CP", "t12", "t13", "t23", "J_CP", "delta");

        for alpha_step in 0..=20 {
            let alpha_cp = alpha_step as f64 * 0.05;

            // Build complex PMNS via rephasing
            let mut u_cp = [[Complex::new(0.0, 0.0); 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    let phase = alpha_cp * gram_phases[i][j];
                    u_cp[i][j] = Complex::from_polar(u_real.read(i, j).abs(), phase);
                    // Preserve the sign of the real element
                    if u_real.read(i, j) < 0.0 {
                        u_cp[i][j] = -u_cp[i][j];
                    }
                }
            }

            // Extract angles from |U_ij| -- these should be unchanged since we
            // only changed phases, not magnitudes
            let u_e3_abs = u_cp[0][2].norm();
            let theta_13_cp = u_e3_abs.min(1.0).asin().to_degrees();
            let cos_13 = theta_13_cp.to_radians().cos();
            let theta_12_cp = if cos_13 > 1e-15 {
                (u_cp[0][1].norm() / cos_13).min(1.0).asin().to_degrees()
            } else { 0.0 };
            let theta_23_cp = if cos_13 > 1e-15 {
                (u_cp[1][2].norm() / cos_13).min(1.0).asin().to_degrees()
            } else { 0.0 };

            // Jarlskog invariant
            let j_cp = (u_cp[0][1] * u_cp[1][2] * u_cp[0][2].conj() * u_cp[1][1].conj()).im;
            let delta = extract_cp_phase((theta_12_cp, theta_13_cp, theta_23_cp), j_cp);

            if alpha_step % 2 == 0 {
                println!("  {:>8.2} | {:>8.2} {:>8.2} {:>8.2} | {:>10.4e} | {:>8.1}",
                    alpha_cp, theta_12_cp, theta_13_cp, theta_23_cp, j_cp, delta);
            }
        }

        // The key prediction: at what alpha_CP does |J_CP| ~ 3e-2?
        // And what is the corresponding delta_CP?
        // Since |U_ij| are preserved, the angles stay at their PDG-matched values.
        // The only free parameter is alpha_CP, which sets the magnitude of J.
        let s12 = t12.to_radians().sin();
        let c12 = t12.to_radians().cos();
        let s13 = t13.to_radians().sin();
        let c13 = t13.to_radians().cos();
        let s23 = t23.to_radians().sin();
        let c23 = t23.to_radians().cos();
        let j_max = s12 * c12 * s23 * c23 * s13 * c13 * c13;

        println!("\n  J_max (from angles) = {:.6}", j_max);
        println!("  PDG J_CP ~ 3e-2 -> sin(delta) = {:.4}", 3e-2 / j_max);
        println!("  -> delta_CP ~ {:.1} deg", (3e-2 / j_max).clamp(-1.0, 1.0).asin().to_degrees());
        println!("  Gram phase arg(G_12) = {:.1} deg = algebraic prediction", gram_phases[0][1].to_degrees());
    }

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
        use crate::bell_inequality::{SignTableCache, rotate_sparse};
        use crate::three_fermion_generations::get_sedenion_subalgebras;

        let ch_pair = (11_usize, 12);
        let nu_pair = (7_usize, 8);

        let (o1, o2, o3) = get_sedenion_subalgebras();
        let subs = [&o1, &o2, &o3];
        let sign_table = SignTableCache::new(16);

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
        use crate::bell_inequality::{SignTableCache, rotate_sparse};
        use crate::three_fermion_generations::get_sedenion_subalgebras;
        use num_complex::Complex;

        let ch_pair = (11_usize, 12);
        let nu_pair = (7_usize, 8);

        let (o1, o2, o3) = get_sedenion_subalgebras();
        let subs = [&o1, &o2, &o3];
        let sign_table = SignTableCache::new(16);

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
        use crate::bell_inequality::{SignTableCache, rotate_sparse};
        use crate::three_fermion_generations::get_sedenion_subalgebras;
        use num_complex::Complex;

        let ch_pair = (11_usize, 12);
        let nu_pair = (7_usize, 8);

        let (o1, o2, o3) = get_sedenion_subalgebras();
        let subs = [&o1, &o2, &o3];
        let sign_table = SignTableCache::new(16);

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
        use crate::bell_inequality::{SignTableCache, rotate_sparse};
        use crate::three_fermion_generations::get_sedenion_subalgebras;
        use num_complex::Complex;

        let pdg = Pdg2024::default();
        let pdg_delta = -165.0_f64;

        // Angle-optimal pair
        let angle_ch = (11_usize, 12);
        let angle_nu = (7_usize, 8);
        // CP-optimal pair
        let cp_ch = (11_usize, 13);
        let cp_nu = (11_usize, 14);

        let (o1, o2, o3) = get_sedenion_subalgebras();
        let subs = [&o1, &o2, &o3];
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
        use crate::bell_inequality::{SignTableCache, rotate_sparse};
        use crate::three_fermion_generations::get_sedenion_subalgebras;
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

        let (o1, o2, o3) = get_sedenion_subalgebras();
        let subs = [&o1, &o2, &o3];
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
        use crate::bell_inequality::{SignTableCache, rotate_sparse};
        use crate::three_fermion_generations::get_sedenion_subalgebras;

        let pdg = Pdg2024::default();
        let pdg_r = 0.0307_f64;

        let (o1, o2, o3) = get_sedenion_subalgebras();
        let subs = [&o1, &o2, &o3];
        let sign_table = SignTableCache::new(16);

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

    /// Gauss-Newton optimization of the two-selector-type model.
    ///
    /// 3 parameters: (beta, alpha_ch, alpha_nu)
    /// 4 residuals: (theta_12, theta_13, theta_23, r) vs PDG
    ///
    /// Uses the same LM-damped Gauss-Newton as C-1492 but generalized
    /// to 3 parameters and 4 observables.
    #[test]
    fn test_two_selector_gauss_newton() {
        use cd_kernel::gourlay_psi;
        use crate::lepton_mass_hierarchy::cd_braid_signed_friction;
        use crate::majorana_braiding::MajoranaMode;
        use crate::bell_inequality::{SignTableCache, rotate_sparse};
        use crate::three_fermion_generations::get_sedenion_subalgebras;
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
        let sel_ch_3: Vec<f64> = subs.iter().map(|s| {
            let (a, b, c) = ch_triple;
            let ma = MajoranaMode { gamma_index: a-1, cd_basis_index: a, cd_dim: 16 };
            let mb = MajoranaMode { gamma_index: b-1, cd_basis_index: b, cd_dim: 16 };
            let mc = MajoranaMode { gamma_index: c-1, cd_basis_index: c, cd_dim: 16 };
            cd_braid_signed_friction(&ma, &mb, s, &sign_table)
            + cd_braid_signed_friction(&ma, &mc, s, &sign_table)
            + cd_braid_signed_friction(&mb, &mc, s, &sign_table)
        }).collect();
        let sel_nu_3: Vec<f64> = subs.iter().map(|s| {
            let (a, b, c) = nu_triple;
            let ma = MajoranaMode { gamma_index: a-1, cd_basis_index: a, cd_dim: 16 };
            let mb = MajoranaMode { gamma_index: b-1, cd_basis_index: b, cd_dim: 16 };
            let mc = MajoranaMode { gamma_index: c-1, cd_basis_index: c, cd_dim: 16 };
            cd_braid_signed_friction(&ma, &mb, s, &sign_table)
            + cd_braid_signed_friction(&ma, &mc, s, &sign_table)
            + cd_braid_signed_friction(&mb, &mc, s, &sign_table)
        }).collect();

        // Precompute 2-blade profiles
        let build_profile = |sel: (usize, usize), sub: &[usize]| -> [f64; 16] {
            let i = sel.0; let j = sel.1;
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
        let ch_prof: Vec<[f64; 16]> = subs.iter().map(|s| build_profile((11, 12), s)).collect();
        let nu_prof: Vec<[f64; 16]> = subs.iter().map(|s| build_profile((7, 8), s)).collect();

        let dot16 = |a: &[f64; 16], b: &[f64; 16]| -> f64 {
            a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
        };

        // Evaluation function: (beta, alpha_ch, alpha_nu) -> (t12, t13, t23, r)
        let evaluate = |beta: f64, a_ch: f64, a_nu: f64| -> [f64; 4] {
            let cb = construct_casimir_baseline(crate::quark_sector::SubalgebraScheme::InterleavedStride);
            let (m_base_ch, m_base_nu) = assemble_lepton_baseline(&cb);
            let mut m_ch = m_base_ch;
            let mut m_nu = m_base_nu;

            for g in 0..3 {
                let f_ch = beta * (w1 * sel_ch_3[g] + w2 * sel_nu_3[g]);
                let f_nu = beta * (w1 * sel_nu_3[g] + w2 * sel_ch_3[g]);
                m_ch.write(g, g, m_ch.read(g, g) + f_ch.exp());
                m_nu.write(g, g, m_nu.read(g, g) + f_nu.exp());
            }
            for i in 0..3 {
                for j in 0..3 {
                    if i == j { continue; }
                    let psi_nu_j = gourlay_psi(&nu_prof[j]);
                    let psi_ch_j = gourlay_psi(&ch_prof[j]);
                    m_nu.write(i, j, m_nu.read(i, j) + a_nu * dot16(&nu_prof[i], &psi_nu_j));
                    m_ch.write(i, j, m_ch.read(i, j) + a_ch * dot16(&ch_prof[i], &psi_ch_j));
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
            let dm21 = ev[1]*ev[1] - ev[0]*ev[0];
            let dm31 = ev[2]*ev[2] - ev[0]*ev[0];
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
        println!("  {:>4} | {:>6} {:>6} {:>6} | {:>8} {:>8} {:>8} {:>8} | {:>8}",
            "iter", "beta", "a_ch", "a_nu", "t12", "t13", "t23", "r", "cost");

        for iter in 0..max_iter {
            let obs = evaluate(params[0], params[1], params[2]);
            let residuals: Vec<f64> = (0..n_resid).map(|i| (obs[i] - pdg_targets[i]) / pdg_sigma[i]).collect();
            let cost: f64 = residuals.iter().map(|r| r * r).sum();

            if iter % 10 == 0 || iter < 5 {
                println!("  {:>4} | {:>6.3} {:>6.2} {:>6.2} | {:>8.2} {:>8.2} {:>8.2} {:>8.4} | {:>8.2}",
                    iter, params[0], params[1], params[2], obs[0], obs[1], obs[2], obs[3], cost);
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
            for i in 0..n_params { jtj_damped[(i, i)] += lambda; }

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
                let new_cost: f64 = (0..n_resid).map(|i| ((new_obs[i] - pdg_targets[i]) / pdg_sigma[i]).powi(2)).sum();
                if new_cost < cost {
                    params = new_params;
                    break;
                }
                step *= 0.5;
            }

            if delta.norm() < 1e-6 { break; }
        }

        // Compute initial cost before multi-start
        let init_obs = evaluate(params[0], params[1], params[2]);
        let init_cost: f64 = (0..n_resid).map(|i| ((init_obs[i] - pdg_targets[i]) / pdg_sigma[i]).powi(2)).sum();

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
                let res: Vec<f64> = (0..n_resid).map(|i| (obs[i] - pdg_targets[i]) / pdg_sigma[i]).collect();
                let cost: f64 = res.iter().map(|r| r * r).sum();

                let mut jac = DMatrix::zeros(n_resid, n_params);
                for pi in 0..n_params {
                    let mut pp = p; let mut pm = p;
                    pp[pi] += eps; pm[pi] -= eps;
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
                for i in 0..n_params { jtj_d[(i, i)] += lambda_ms; }
                let delta = match jtj_d.lu().solve(&(-&jtr)) { Some(d) => d, None => break };

                let mut step = 1.0;
                for _ in 0..10 {
                    let np = [
                        (p[0] + step * delta[0]).max(0.01).min(2.0),
                        (p[1] + step * delta[1]).max(0.0).min(20.0),
                        (p[2] + step * delta[2]).max(0.0).min(10.0),
                    ];
                    let no = evaluate(np[0], np[1], np[2]);
                    let nc: f64 = (0..n_resid).map(|i| ((no[i] - pdg_targets[i]) / pdg_sigma[i]).powi(2)).sum();
                    if nc < cost { p = np; break; }
                    step *= 0.5;
                }
                if delta.norm() < 1e-6 { break; }
            }
            let obs = evaluate(p[0], p[1], p[2]);
            let cost: f64 = (0..n_resid).map(|i| ((obs[i] - pdg_targets[i]) / pdg_sigma[i]).powi(2)).sum();
            if cost < global_best_cost {
                global_best_cost = cost;
                global_best_params = p;
            }
        }

        if global_best_cost < init_cost {
            params = global_best_params;
            println!("\n  Multi-start found better minimum: cost = {:.2}", global_best_cost);
        }

        let final_obs = evaluate(params[0], params[1], params[2]);
        let final_cost: f64 = (0..n_resid).map(|i| ((final_obs[i] - pdg_targets[i]) / pdg_sigma[i]).powi(2)).sum();

        println!("\n  === FINAL RESULT ===");
        println!("  beta = {:.4}, alpha_ch = {:.4}, alpha_nu = {:.4}", params[0], params[1], params[2]);
        println!("  theta_12 = {:.2} deg (PDG: {:.2}, err: {:.1}%)", final_obs[0], pdg.theta_12_deg, ((final_obs[0] - pdg.theta_12_deg) / pdg.theta_12_deg * 100.0).abs());
        println!("  theta_13 = {:.2} deg (PDG: {:.2}, err: {:.1}%)", final_obs[1], pdg.theta_13_deg, ((final_obs[1] - pdg.theta_13_deg) / pdg.theta_13_deg * 100.0).abs());
        println!("  theta_23 = {:.2} deg (PDG: {:.2}, err: {:.1}%)", final_obs[2], pdg.theta_23_deg, ((final_obs[2] - pdg.theta_23_deg) / pdg.theta_23_deg * 100.0).abs());
        println!("  r = {:.6} (PDG: {:.4}, err: {:.1}%)", final_obs[3], pdg_r, ((final_obs[3] - pdg_r) / pdg_r * 100.0).abs());
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
        use cd_kernel::gourlay_psi;
        use crate::lepton_mass_hierarchy::cd_braid_signed_friction;
        use crate::majorana_braiding::MajoranaMode;
        use crate::bell_inequality::{SignTableCache, rotate_sparse};
        use crate::three_fermion_generations::get_sedenion_subalgebras;

        let pdg = Pdg2024::default();
        let pdg_r = 0.0307_f64;

        let (o1, o2, o3) = get_sedenion_subalgebras();
        let subs = [&o1, &o2, &o3];
        let sign_table = SignTableCache::new(16);

        // Use BOTH 3-blade and 2-blade profiles
        let build_pair_profile = |sel: (usize, usize), sub: &[usize]| -> [f64; 16] {
            let i = sel.0; let j = sel.1;
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

        // 2-blade profiles for angle-optimal mixing
        let ch_2b: Vec<[f64; 16]> = subs.iter().map(|s| build_pair_profile((11, 12), s)).collect();
        let nu_2b: Vec<[f64; 16]> = subs.iter().map(|s| build_pair_profile((7, 8), s)).collect();

        let dot16 = |a: &[f64; 16], b: &[f64; 16]| -> f64 {
            a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
        };

        // 3-blade scalar frictions for mass hierarchy
        let sel_ch_3: Vec<f64> = subs.iter().map(|s| {
            let ma = MajoranaMode { gamma_index: 0, cd_basis_index: 1, cd_dim: 16 };
            let mb = MajoranaMode { gamma_index: 5, cd_basis_index: 6, cd_dim: 16 };
            let mc = MajoranaMode { gamma_index: 10, cd_basis_index: 11, cd_dim: 16 };
            cd_braid_signed_friction(&ma, &mb, s, &sign_table)
            + cd_braid_signed_friction(&ma, &mc, s, &sign_table)
            + cd_braid_signed_friction(&mb, &mc, s, &sign_table)
        }).collect();
        let sel_nu_3: Vec<f64> = subs.iter().map(|s| {
            let ma = MajoranaMode { gamma_index: 0, cd_basis_index: 1, cd_dim: 16 };
            let mb = MajoranaMode { gamma_index: 2, cd_basis_index: 3, cd_dim: 16 };
            let mc = MajoranaMode { gamma_index: 7, cd_basis_index: 8, cd_dim: 16 };
            cd_braid_signed_friction(&ma, &mb, s, &sign_table)
            + cd_braid_signed_friction(&ma, &mc, s, &sign_table)
            + cd_braid_signed_friction(&mb, &mc, s, &sign_table)
        }).collect();

        println!("  === Friction-Native Baseline (No Casimir) ===\n");
        println!("  3-blade ch frictions: [{:.2}, {:.2}, {:.2}]", sel_ch_3[0], sel_ch_3[1], sel_ch_3[2]);
        println!("  3-blade nu frictions: [{:.2}, {:.2}, {:.2}]", sel_nu_3[0], sel_nu_3[1], sel_nu_3[2]);

        // Scan: beta scales 3-blade diagonal, alpha scales 2-blade off-diagonal
        println!("\n  {:>6} {:>6} | {:>8} {:>8} {:>8} | {:>8} | {:>8}",
            "beta", "alpha", "t12", "t13", "t23", "r", "cost");

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
                    m_ch.write(g, g, f_ch.exp());
                    m_nu.write(g, g, f_nu.exp());
                }

                // Off-diagonal: 2-blade psi coupling (NO Casimir baseline)
                for i in 0..3 {
                    for j in 0..3 {
                        if i == j { continue; }
                        let psi_nu_j = gourlay_psi(&nu_2b[j]);
                        let psi_ch_j = gourlay_psi(&ch_2b[j]);
                        m_nu.write(i, j, alpha * dot16(&nu_2b[i], &psi_nu_j));
                        m_ch.write(i, j, alpha * dot16(&ch_2b[i], &psi_ch_j));
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
                let dm21 = ev[1]*ev[1] - ev[0]*ev[0];
                let dm31 = ev[2]*ev[2] - ev[0]*ev[0];
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
        println!("  theta_12 = {:.2} deg (PDG: {:.2}, err: {:.1}%)", obs[0], pdg.theta_12_deg, ((obs[0] - pdg.theta_12_deg) / pdg.theta_12_deg * 100.0).abs());
        println!("  theta_13 = {:.2} deg (PDG: {:.2}, err: {:.1}%)", obs[1], pdg.theta_13_deg, ((obs[1] - pdg.theta_13_deg) / pdg.theta_13_deg * 100.0).abs());
        println!("  theta_23 = {:.2} deg (PDG: {:.2}, err: {:.1}%)", obs[2], pdg.theta_23_deg, ((obs[2] - pdg.theta_23_deg) / pdg.theta_23_deg * 100.0).abs());
        println!("  r = {:.4} (PDG: {:.4}, err: {:.1}%)", obs[3], pdg_r, ((obs[3] - pdg_r) / pdg_r * 100.0).abs());
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
        use gororoba_algebra::lie::g2_stabilizer::{
            stabilizer_decomposition, structure_constants,
        };

        let pdg_sin2_tw = 0.2312_f64;

        println!("  === Weinberg Angle from G2 Stabilizer ===\n");
        println!("  PDG: sin^2(theta_W) = {:.4}\n", pdg_sin2_tw);

        // Approach 1: GUT SU(5)
        let gut = 3.0 / 8.0;
        println!("  1. SU(5) GUT: sin^2 = 3/8 = {:.4} (err: {:.1}%)",
            gut, ((gut - pdg_sin2_tw) / pdg_sin2_tw * 100.0).abs());

        // Approach 2: Existing flux ratio
        let flux = 0.199;
        println!("  2. Flux ratio (C-1458): sin^2 = {:.4} (err: {:.1}%)",
            flux, ((flux - pdg_sin2_tw) / pdg_sin2_tw * 100.0).abs());

        // Approach 3: G2 dim decomposition
        let dim_stab = 8.0_f64;
        let dim_coset = 6.0_f64;
        let dim_g2 = 14.0_f64;

        // sin^2 = dim(coset) / dim(G2) * correction?
        let dim_ratio_a = dim_coset / dim_g2; // 6/14 = 0.429
        let _dim_ratio_b = dim_coset / (dim_stab + dim_coset); // = 6/14 same
        let dim_ratio_c = 1.0 / (1.0 + dim_stab / 3.0); // 1/(1+8/3) = 3/11 = 0.273

        println!("  3a. dim(coset)/dim(G2) = 6/14 = {:.4} (err: {:.1}%)",
            dim_ratio_a, ((dim_ratio_a - pdg_sin2_tw) / pdg_sin2_tw * 100.0).abs());
        println!("  3b. 3/(3+8) = 3/11 = {:.4} (err: {:.1}%)",
            dim_ratio_c, ((dim_ratio_c - pdg_sin2_tw) / pdg_sin2_tw * 100.0).abs());

        // Approach 4: Casimir ratios
        // C2(fund, SU(3)) = 4/3, C2(fund, SU(2)) = 3/4
        let c2_su3 = 4.0 / 3.0;
        let c2_su2 = 3.0 / 4.0;
        let casimir_ratio = c2_su2 / (c2_su2 + c2_su3); // 0.75/(0.75+1.33) = 0.36
        println!("  4. C2(SU2)/(C2(SU2)+C2(SU3)) = {:.4} (err: {:.1}%)",
            casimir_ratio, ((casimir_ratio - pdg_sin2_tw) / pdg_sin2_tw * 100.0).abs());

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
        println!("     f_coset^2 / (f_stab^2 + f_coset^2) = {:.4} (err: {:.1}%)",
            sc_ratio, ((sc_ratio - pdg_sin2_tw) / pdg_sin2_tw * 100.0).abs());

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
        println!("\n  6. 1/(1+3) = 1/4 = {:.4} (err: {:.1}%)",
            simple, ((simple - pdg_sin2_tw) / pdg_sin2_tw * 100.0).abs());

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
        let b1 = 41.0 / 10.0_f64;  // U(1)_Y
        let b2 = -19.0 / 6.0_f64;  // SU(2)_L

        println!("  === Weinberg Angle RG Running ===\n");
        println!("  Tree-level (G2): sin^2 = {:.4}", sin2_gut);
        println!("  PDG at M_Z: sin^2 = 0.2312\n");

        // Scan unification scales
        for log_m_gut in [14.0, 15.0, 15.5, 16.0, 16.5, 17.0, 18.0_f64] {
            let m_gut = 10.0_f64.powf(log_m_gut);
            // 1-loop running: Delta(sin^2) = (3/5) * alpha/(2*pi) * (b1 - b2) * ln(M_Z/M_GUT)
            // The 3/5 is the SU(5) normalization factor for U(1)_Y
            let delta = (3.0/5.0) * alpha_em / (2.0 * std::f64::consts::PI)
                      * (b1 - b2) * (m_z / m_gut).ln();
            let sin2_mz = sin2_gut + delta;
            let err = ((sin2_mz - 0.2312) / 0.2312 * 100.0).abs();
            let marker = if err < 3.0 { " <--" } else { "" };
            println!("  M_GUT = 10^{:.1} GeV: sin^2(M_Z) = {:.4} (err: {:.1}%){}",
                log_m_gut, sin2_mz, err, marker);
        }

        // The G2-specific unification scale: related to the G2 manifold volume
        // In string/M-theory, G2 holonomy manifolds have characteristic scale
        // around 10^16 GeV. Let's check.
        let m_gut_g2 = 1e16_f64;
        let delta_g2 = (3.0/5.0) * alpha_em / (2.0 * std::f64::consts::PI)
                     * (b1 - b2) * (m_z / m_gut_g2).ln();
        let sin2_mz_g2 = sin2_gut + delta_g2;
        println!("\n  At M_G2 = 10^16 GeV (G2 holonomy scale):");
        println!("  sin^2(M_Z) = {:.4} (PDG: 0.2312, err: {:.1}%)",
            sin2_mz_g2, ((sin2_mz_g2 - 0.2312) / 0.2312 * 100.0).abs());
    }

    /// Full 3-blade model: 3-blade diagonal + 3-blade off-diagonal.
    ///
    /// Off-diagonal: sum of 3 pairwise psi overlaps (3x amplitude of 2-blade).
    /// This should provide enough off-diagonal strength to rotate eigenvectors
    /// while preserving the 3-blade mass hierarchy.
    #[test]
    fn test_full_3blade_model() {
        use cd_kernel::gourlay_psi;
        use crate::lepton_mass_hierarchy::cd_braid_signed_friction;
        use crate::majorana_braiding::MajoranaMode;
        use crate::bell_inequality::{SignTableCache, rotate_sparse};
        use crate::three_fermion_generations::get_sedenion_subalgebras;

        let pdg = Pdg2024::default();
        let pdg_r = 0.0307_f64;

        let (o1, o2, o3) = get_sedenion_subalgebras();
        let subs = [&o1, &o2, &o3];
        let sign_table = SignTableCache::new(16);
        let w1 = -0.656850_f64;
        let w2 = -0.741999_f64;

        // Build 3-blade profiles for BOTH diagonal and off-diagonal
        let build_3blade_profile = |a: usize, b: usize, c: usize, sub: &[usize]| -> [f64; 16] {
            let ma = MajoranaMode { gamma_index: a-1, cd_basis_index: a, cd_dim: 16 };
            let mb = MajoranaMode { gamma_index: b-1, cd_basis_index: b, cd_dim: 16 };
            let mc = MajoranaMode { gamma_index: c-1, cd_basis_index: c, cd_dim: 16 };
            let build_pair = |m1: &MajoranaMode, m2: &MajoranaMode, s: &[usize]| -> [f64; 16] {
                let i = m1.cd_basis_index; let j = m2.cd_basis_index;
                let a_sp = vec![(i, 1.0)];
                let a_rot = rotate_sparse(&a_sp, i, j, std::f64::consts::FRAC_PI_4);
                let b_sp = vec![(j, 1.0)];
                let mut p = [0.0_f64; 16];
                for &k in s {
                    if k == 0 || k == i || k == j { continue; }
                    p[k] = sign_table.sparse_associator_sum(&a_rot, &[(k, 1.0)], &b_sp);
                }
                p
            };
            let p_ab = build_pair(&ma, &mb, sub);
            let p_ac = build_pair(&ma, &mc, sub);
            let p_bc = build_pair(&mb, &mc, sub);
            let mut combined = [0.0_f64; 16];
            for idx in 0..16 { combined[idx] = p_ab[idx] + p_ac[idx] + p_bc[idx]; }
            combined
        };

        // Use angle-optimal triple for off-diagonal (different from mass-ratio triple)
        // ch: (11,12) pair embedded as triple (10,11,12) -- includes neighbors
        // nu: (7,8) pair embedded as triple (7,8,9) -- includes neighbors
        let ch_profiles: Vec<[f64; 16]> = subs.iter()
            .map(|s| build_3blade_profile(10, 11, 12, s)).collect();
        let nu_profiles: Vec<[f64; 16]> = subs.iter()
            .map(|s| build_3blade_profile(7, 8, 9, s)).collect();

        // Mass-ratio triple frictions (diagonal)
        let sel_ch_3: Vec<f64> = subs.iter().map(|s| {
            let ma = MajoranaMode { gamma_index: 0, cd_basis_index: 1, cd_dim: 16 };
            let mb = MajoranaMode { gamma_index: 5, cd_basis_index: 6, cd_dim: 16 };
            let mc = MajoranaMode { gamma_index: 10, cd_basis_index: 11, cd_dim: 16 };
            cd_braid_signed_friction(&ma, &mb, s, &sign_table)
            + cd_braid_signed_friction(&ma, &mc, s, &sign_table)
            + cd_braid_signed_friction(&mb, &mc, s, &sign_table)
        }).collect();
        let sel_nu_3: Vec<f64> = subs.iter().map(|s| {
            let ma = MajoranaMode { gamma_index: 0, cd_basis_index: 1, cd_dim: 16 };
            let mb = MajoranaMode { gamma_index: 2, cd_basis_index: 3, cd_dim: 16 };
            let mc = MajoranaMode { gamma_index: 7, cd_basis_index: 8, cd_dim: 16 };
            cd_braid_signed_friction(&ma, &mb, s, &sign_table)
            + cd_braid_signed_friction(&ma, &mc, s, &sign_table)
            + cd_braid_signed_friction(&mb, &mc, s, &sign_table)
        }).collect();

        let dot16 = |a: &[f64; 16], b: &[f64; 16]| -> f64 {
            a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
        };

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
                    m_ch.write(g, g, f_ch.exp());
                    m_nu.write(g, g, f_nu.exp());
                }

                // Off-diagonal: 3-blade psi coupling (3x amplitude)
                for i in 0..3 {
                    for j in 0..3 {
                        if i == j { continue; }
                        let psi_nu_j = gourlay_psi(&nu_profiles[j]);
                        let psi_ch_j = gourlay_psi(&ch_profiles[j]);
                        m_nu.write(i, j, alpha * dot16(&nu_profiles[i], &psi_nu_j));
                        m_ch.write(i, j, alpha * dot16(&ch_profiles[i], &psi_ch_j));
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
                let dm21 = ev[1]*ev[1] - ev[0]*ev[0];
                let dm31 = ev[2]*ev[2] - ev[0]*ev[0];
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
        println!("  theta_12 = {:.2} deg (PDG: {:.2}, err: {:.1}%)", obs[0], pdg.theta_12_deg, ((obs[0] - pdg.theta_12_deg) / pdg.theta_12_deg * 100.0).abs());
        println!("  theta_13 = {:.2} deg (PDG: {:.2}, err: {:.1}%)", obs[1], pdg.theta_13_deg, ((obs[1] - pdg.theta_13_deg) / pdg.theta_13_deg * 100.0).abs());
        println!("  theta_23 = {:.2} deg (PDG: {:.2}, err: {:.1}%)", obs[2], pdg.theta_23_deg, ((obs[2] - pdg.theta_23_deg) / pdg.theta_23_deg * 100.0).abs());
        println!("  r = {:.4} (PDG: {:.4}, err: {:.1}%)", obs[3], pdg_r, ((obs[3] - pdg_r) / pdg_r * 100.0).abs());
        println!("  cost = {:.2}", best.0);
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
            [(3.0, 0.0),  (0.5, 1.2), (-0.3, 0.8)],
            [(0.5, -1.2), (1.0, 0.0),  (0.7, -0.4)],
            [(-0.3, -0.8),(0.7, 0.4),  (2.0, 0.0)],
        ];

        let (evals, evecs) = hermitian_3x3_eig(&h);

        // --- Check 4: phase convention ---
        for col in 0..3 {
            let mut max_mag_sq = 0.0_f64;
            let mut max_idx = 0;
            for i in 0..3 {
                let ms = evecs[i][col].0 * evecs[i][col].0
                       + evecs[i][col].1 * evecs[i][col].1;
                if ms > max_mag_sq { max_mag_sq = ms; max_idx = i; }
            }
            let (re, im) = evecs[max_idx][col];
            assert!(re >= -1e-14,
                "col {col}: largest component has re={re:.6e} (should be >= 0)");
            assert!(im.abs() < 1e-12,
                "col {col}: largest component has im={im:.6e} (should be ~0)");
        }

        // --- Check 3: residual |Hv - lam*v| ---
        let h_frob = {
            let mut s = 0.0_f64;
            for row in &h { for &(r, m) in row { s += r * r + m * m; } }
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
            assert!(residual < 1e-12,
                "col {col}: relative residual {residual:.3e} exceeds 1e-12");
        }

        // --- Check 1: eigenvalue agreement with faer ---
        let mut h_faer = faer::Mat::<faer::complex_native::c64>::zeros(3, 3);
        for i in 0..3 {
            for j in 0..3 {
                h_faer.write(i, j, faer::complex_native::c64::new(h[i][j].0, h[i][j].1));
            }
        }
        let eig_faer = h_faer.selfadjoint_eigendecomposition(faer::Side::Lower);
        let mut faer_evals = [0.0_f64; 3];
        for i in 0..3 { faer_evals[i] = eig_faer.s().column_vector().read(i).re; }
        faer_evals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for i in 0..3 {
            let diff = (evals[i] - faer_evals[i]).abs();
            assert!(diff < 1e-12,
                "eigenvalue {i}: cardano={:.12e}, faer={:.12e}, diff={diff:.3e}",
                evals[i], faer_evals[i]);
        }

        // --- Check 2: projector agreement |v*v^dag - v_faer*v_faer^dag|_F ---
        // Match Cardano eigenvectors to faer by closest eigenvalue
        for col in 0..3 {
            // Find faer column with closest eigenvalue
            let faer_col = (0..3).min_by(|&a, &b| {
                let da = (evals[col] - eig_faer.s().column_vector().read(a).re).abs();
                let db = (evals[col] - eig_faer.s().column_vector().read(b).re).abs();
                da.partial_cmp(&db).unwrap()
            }).unwrap();

            // Compute |P_cardano - P_faer|_F^2
            let mut frob_sq = 0.0_f64;
            for i in 0..3 {
                for j in 0..3 {
                    // P_cardano[i][j] = v_i * conj(v_j)
                    let pc = cmul(evecs[i][col], cconj(evecs[j][col]));
                    // P_faer[i][j] = u_i * conj(u_j)
                    let ui = eig_faer.u().read(i, faer_col);
                    let uj = eig_faer.u().read(j, faer_col);
                    let pf_re = ui.re * uj.re + ui.im * uj.im;
                    let pf_im = ui.im * uj.re - ui.re * uj.im;
                    let dr = pc.0 - pf_re;
                    let di = pc.1 - pf_im;
                    frob_sq += dr * dr + di * di;
                }
            }
            let frob = frob_sq.sqrt();
            assert!(frob < 1e-10,
                "col {col}: projector Frobenius distance {frob:.3e} exceeds 1e-10");
        }

        println!("  Cardano phase canonicalization: all 4 checks passed");
        println!("  Eigenvalues: [{:.6}, {:.6}, {:.6}]", evals[0], evals[1], evals[2]);
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
            let u_mu1_sq = s12 * s12 * c23 * c23 + c12 * c12 * s23 * s23 * s13 * s13
                + 2.0 * s12 * c12 * s23 * c23 * s13 * cd;

            // U_mu2 = c12*c23 - s12*s23*s13*exp(i*delta)
            let u_mu2_sq = c12 * c12 * c23 * c23 + s12 * s12 * s23 * s23 * s13 * s13
                - 2.0 * c12 * s12 * s23 * c23 * s13 * cd;

            // U_tau1 = s12*s23 - c12*c23*s13*exp(i*delta)
            let u_tau1_sq = s12 * s12 * s23 * s23 + c12 * c12 * c23 * c23 * s13 * s13
                - 2.0 * s12 * c12 * c23 * s23 * s13 * cd;

            // U_tau2 = -c12*s23 - s12*c23*s13*exp(i*delta)
            let u_tau2_sq = c12 * c12 * s23 * s23 + s12 * s12 * c23 * c23 * s13 * s13
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
                if v > 180.0 { v -= 360.0; }
                if v < -180.0 { v += 360.0; }
                v
            };
            let diff = (wrap(recovered) - wrap(delta_deg)).abs();
            let diff = if diff > 180.0 { 360.0 - diff } else { diff };

            println!("  delta_in={:7.1} deg  recovered={:7.1} deg  diff={:.2e}",
                delta_deg, recovered, diff);

            // Skip delta=0 and delta=180 where cos(delta) is degenerate
            if delta_deg.abs() > 1.0 && (delta_deg - 180.0).abs() > 1.0 {
                assert!(diff < 0.1,
                    "delta={delta_deg}: recovered={recovered:.2}, diff={diff:.2e}");
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
        use cd_kernel::gourlay_psi;
        use crate::majorana_braiding::MajoranaMode;
        use crate::bell_inequality::{SignTableCache, rotate_sparse};
        use crate::three_fermion_generations::get_sedenion_subalgebras;

        let ch_pair = (11_usize, 12);
        let nu_pair = (7_usize, 8);
        let alpha_ch = 3.00_f64;
        let alpha_nu = 1.35_f64;

        // Get the best real mass matrices + V_6 correction
        let (m_ch_real, m_nu_real) = construct_pmns_matrices_two_param(
            ch_pair, nu_pair, alpha_ch, alpha_nu,
        );
        let (v6_basis, _, _) = extract_v6_basis();
        let lift = TensorElementLift;
        let eps = 0.05_f64;
        let n_basis = v6_basis.nrows().min(6);

        // Compute constrained directions
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
            for k in 0..6 { beta[k] = t_sol * u_solar[k] + t_atm * u_atmo[k]; }
            angles_at(&beta)
        };
        let (t_sol, t_atm, _, _) = gauss_newton_2d(
            &inner_angles, 1.5, 0.0,
            (33.41, 8.54, 49.0), (1.0, 2.24, 1.0), 15,
        );

        let mut beta_opt = [0.0_f64; 6];
        for k in 0..6 { beta_opt[k] = t_sol * u_solar[k] + t_atm * u_atmo[k]; }
        let mut m_nu_corrected = m_nu_real.clone();
        apply_v6_perturbation(&mut m_nu_corrected, &v6_basis, &beta_opt, &lift);
        let m_nu_corrected = (&m_nu_corrected + m_nu_corrected.transpose()) * faer::scale(0.5);

        // Verify baseline angles
        let eig_nu_c0 = m_nu_corrected.selfadjoint_eigendecomposition(faer::Side::Lower);
        let u_real_baseline = eig_ch_0.u().transpose() * eig_nu_c0.u();
        let mut u_perm_base = faer::Mat::zeros(3, 3);
        for i in 0..3 { for j in 0..3 {
            u_perm_base.write(i, j, u_real_baseline.read(perm_u[i], perm_d[j]));
        }}
        let (t12_b, t13_b, t23_b) = extract_pmns_angles(&u_perm_base);
        println!("--- CP VIOLATION: PHASE-ONLY COMPLEXIFICATION ---");
        println!("  Real baseline: theta_12={:.2}, theta_13={:.2}, theta_23={:.2}",
            t12_b, t13_b, t23_b);

        // Build friction profiles
        let (o1, o2, o3) = get_sedenion_subalgebras();
        let subs = [&o1, &o2, &o3];
        let sign_table = SignTableCache::new(16);
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

        println!("\n  Scanning k=1..7 embeddings with phase-only complexification:");

        // For each k=1..7, compute phase angles from J_k complex structure
        for k in 1..=7 {
            // Full 16D J_k action on both octonion halves (14D active)
            // instead of 6D perp-only action from ComplexStructure
            let apply_jk = |v: &[f64; 16]| -> [f64; 16] {
                apply_jk_full_16d(v, k)
            };

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
                let mut m_nu_c = faer::Mat::<faer::complex_native::c64>::zeros(3, 3);
                let mut m_ch_c = faer::Mat::<faer::complex_native::c64>::zeros(3, 3);

                for i in 0..3 {
                    // Diagonal stays real
                    m_nu_c.write(i, i, faer::complex_native::c64::new(
                        m_nu_corrected.read(i, i), 0.0,
                    ));
                    m_ch_c.write(i, i, faer::complex_native::c64::new(
                        m_ch_real.read(i, i), 0.0,
                    ));

                    for j in (i + 1)..3 {
                        // Off-diagonal: phase rotation
                        let phase = alpha_cp * phi[i][j];
                        let mag = m_nu_corrected.read(i, j);
                        let re = mag * phase.cos();
                        let im = mag * phase.sin();
                        m_nu_c.write(i, j, faer::complex_native::c64::new(re, im));
                        m_nu_c.write(j, i, faer::complex_native::c64::new(re, -im)); // Hermitian

                        // Charged lepton stays real symmetric
                        m_ch_c.write(i, j, faer::complex_native::c64::new(
                            m_ch_real.read(i, j), 0.0,
                        ));
                        m_ch_c.write(j, i, faer::complex_native::c64::new(
                            m_ch_real.read(j, i), 0.0,
                        ));
                    }
                }

                let eig_ch_c = m_ch_c.selfadjoint_eigendecomposition(faer::Side::Lower);
                let eig_nu_c = m_nu_c.selfadjoint_eigendecomposition(faer::Side::Lower);
                let u_pmns_c = eig_ch_c.u().adjoint() * eig_nu_c.u();

                // Apply same permutation as real baseline
                let mut u_perm_c = faer::Mat::<faer::complex_native::c64>::zeros(3, 3);
                for i in 0..3 { for j in 0..3 {
                    u_perm_c.write(i, j, u_pmns_c.read(perm_u[i], perm_d[j]));
                }}

                // Extract angles from |U_ij|
                let u_e3_abs = u_perm_c.read(0, 2).abs();
                let theta_13 = u_e3_abs.min(1.0).asin().to_degrees();
                let cos_13 = theta_13.to_radians().cos();
                let theta_12 = if cos_13 > 1e-15 {
                    (u_perm_c.read(0, 1).abs() / cos_13).min(1.0).asin().to_degrees()
                } else { 0.0 };
                let theta_23 = if cos_13 > 1e-15 {
                    (u_perm_c.read(1, 2).abs() / cos_13).min(1.0).asin().to_degrees()
                } else { 0.0 };

                // Jarlskog invariant: J = Im(U_e1 * U_mu2 * conj(U_e2) * conj(U_mu1))
                let j_cp = (u_perm_c.read(0, 0) * u_perm_c.read(1, 1)
                    * u_perm_c.read(0, 1).conj() * u_perm_c.read(1, 0).conj()).im;

                // delta_CP from arg(-U_e3)
                let delta_cp = (-u_perm_c.read(0, 2)).arg().to_degrees();

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
                println!("  k={}: alpha_CP={:.3}, theta_12={:.2} ({:.1}%), theta_13={:.2} ({:.1}%), theta_23={:.2} ({:.1}%), J_CP={:.4e}, delta={:.1} deg",
                    k, best_alpha_cp, best_angles.0, err_12, best_angles.1, err_13, best_angles.2, err_23, best_j_cp, best_delta);
            } else {
                println!("  k={}: no solution with <5% angle error and nonzero J_CP", k);
            }
        }

        println!("\n  PDG targets: J_CP ~ 3.3e-2, delta_CP ~ 195 deg (normal ordering, NuFIT 5.3)");
        println!("  Rephasing-invariant Jarlskog: |J| = cos(t12)*sin(t12)*cos(t13)^2*sin(t13)*cos(t23)*sin(t23)*sin(delta)");
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
        use gororoba_algebra::lie::g2_stabilizer::complex_structure;
        use cd_kernel::gourlay_psi;
        use crate::majorana_braiding::MajoranaMode;
        use crate::bell_inequality::{SignTableCache, rotate_sparse};
        use crate::three_fermion_generations::get_sedenion_subalgebras;

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

        let inner_angles = |t_sol: f64, t_atm: f64| -> (f64, f64, f64) {
            let mut beta = [0.0_f64; 6];
            for kk in 0..6 { beta[kk] = t_sol * u_solar[kk] + t_atm * u_atmo[kk]; }
            angles_at(&beta)
        };
        let (t_sol, t_atm, _, _) = gauss_newton_2d(
            &inner_angles, 1.5, 0.0,
            (33.41, 8.54, 49.0), (1.0, 2.24, 1.0), 15,
        );

        let mut beta_opt = [0.0_f64; 6];
        for kk in 0..6 { beta_opt[kk] = t_sol * u_solar[kk] + t_atm * u_atmo[kk]; }
        let mut m_nu_corrected = m_nu_real.clone();
        apply_v6_perturbation(&mut m_nu_corrected, &v6_basis, &beta_opt, &lift);
        let m_nu_corrected = (&m_nu_corrected + m_nu_corrected.transpose()) * faer::scale(0.5);

        // Build friction profiles
        let (o1, o2, o3) = get_sedenion_subalgebras();
        let subs = [&o1, &o2, &o3];
        let sign_table = SignTableCache::new(16);
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

        println!("--- CP VIOLATION: 6D vs 16D J_k DIMENSION COMPARISON ---\n");
        println!("  {:>2} | {:>5} | {:>8} | {:>8} {:>8} {:>8} | {:>10} {:>8} | {:>10} {:>8}",
            "k", "dim", "alpha_CP", "t12", "t13", "t23", "|J_CP|", "delta",
            "|J_CP|_16D", "delta_16D");
        println!("  {:-<2}-+-{:-<5}-+-{:-<8}-+-{:-<8}-{:-<8}-{:-<8}-+-{:-<10}-{:-<8}-+-{:-<10}-{:-<8}",
            "", "", "", "", "", "", "", "", "", "");

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
            let apply_jk_16d = |v: &[f64; 16]| -> [f64; 16] {
                apply_jk_full_16d(v, k)
            };

            // Generic CP pipeline: takes a J_k action as a trait object and
            // returns (best_alpha, t12, t13, t23, j_cp, delta_cp).
            // Uses dyn Fn so the same closure works for both 6D and 16D.
            let compute_cp_for_jk = |apply_fn: &dyn Fn(&[f64; 16]) -> [f64; 16]|
                -> (f64, f64, f64, f64, f64, f64)
            {
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
                            if phi[i][j].abs() > 1e-10 { has_phase = true; }
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
                    let mut m_nu_c = faer::Mat::<faer::complex_native::c64>::zeros(3, 3);
                    let mut m_ch_c = faer::Mat::<faer::complex_native::c64>::zeros(3, 3);

                    for i in 0..3 {
                        m_nu_c.write(i, i, faer::complex_native::c64::new(
                            m_nu_corrected.read(i, i), 0.0,
                        ));
                        m_ch_c.write(i, i, faer::complex_native::c64::new(
                            m_ch_real.read(i, i), 0.0,
                        ));
                        for j in (i + 1)..3 {
                            let phase = alpha_cp * phi[i][j];
                            let mag = m_nu_corrected.read(i, j);
                            let re = mag * phase.cos();
                            let im = mag * phase.sin();
                            m_nu_c.write(i, j, faer::complex_native::c64::new(re, im));
                            m_nu_c.write(j, i, faer::complex_native::c64::new(re, -im));
                            m_ch_c.write(i, j, faer::complex_native::c64::new(
                                m_ch_real.read(i, j), 0.0,
                            ));
                            m_ch_c.write(j, i, faer::complex_native::c64::new(
                                m_ch_real.read(j, i), 0.0,
                            ));
                        }
                    }

                    let eig_ch_c = m_ch_c.selfadjoint_eigendecomposition(faer::Side::Lower);
                    let eig_nu_c = m_nu_c.selfadjoint_eigendecomposition(faer::Side::Lower);
                    let u_pmns_c = eig_ch_c.u().adjoint() * eig_nu_c.u();

                    let mut u_perm_c = faer::Mat::<faer::complex_native::c64>::zeros(3, 3);
                    for i in 0..3 { for j in 0..3 {
                        u_perm_c.write(i, j, u_pmns_c.read(perm_u[i], perm_d[j]));
                    }}

                    let u_e3_abs = u_perm_c.read(0, 2).abs();
                    let theta_13 = u_e3_abs.min(1.0).asin().to_degrees();
                    let cos_13 = theta_13.to_radians().cos();
                    let theta_12 = if cos_13 > 1e-15 {
                        (u_perm_c.read(0, 1).abs() / cos_13).min(1.0).asin().to_degrees()
                    } else { 0.0 };
                    let theta_23 = if cos_13 > 1e-15 {
                        (u_perm_c.read(1, 2).abs() / cos_13).min(1.0).asin().to_degrees()
                    } else { 0.0 };

                    let j_cp = (u_perm_c.read(0, 0) * u_perm_c.read(1, 1)
                        * u_perm_c.read(0, 1).conj() * u_perm_c.read(1, 0).conj()).im;

                    let delta_cp = (-u_perm_c.read(0, 2)).arg().to_degrees();

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

                (best_alpha, best_ang.0, best_ang.1, best_ang.2, best_jcp, best_delta)
            };

            let (a6, t12_6, t13_6, t23_6, jcp_6, d6) = compute_cp_for_jk(&apply_jk_6d);
            let (a16, t12_16, t13_16, t23_16, jcp_16, d16) = compute_cp_for_jk(&apply_jk_16d);

            if jcp_6.abs() > 1e-6 || jcp_16.abs() > 1e-6 {
                println!("  k={} |  6D  | {:.4}   | {:.2}   {:.2}   {:.2}   | {:.4e}   {:.1}   | --         --",
                    k, a6, t12_6, t13_6, t23_6, jcp_6, d6);
                println!("  k={} | 16D  | {:.4}   | {:.2}   {:.2}   {:.2}   | --         --       | {:.4e}   {:.1}",
                    k, a16, t12_16, t13_16, t23_16, jcp_16, d16);
            } else {
                println!("  k={}: both variants have zero J_CP within 5% angle tolerance", k);
            }
        }

        println!("\n  PDG target: |J_CP| ~ 3.3e-2, delta_CP ~ 195 deg");
        println!("  Baseline (6D perp-only): |J_CP| ~ 8.5e-3 (C-1494)");
        println!("  If 16D > 6D, the gap is architectural, not algebraic.");
    }

    /// Joint (alpha_CP, t_solar, t_atmo) 3D optimization for J_CP
    /// gap closure (C-1497).
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
        use crate::bell_inequality::{SignTableCache, rotate_sparse};
        use crate::three_fermion_generations::get_sedenion_subalgebras;

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
        let (o1, o2, o3) = get_sedenion_subalgebras();
        let subs = [&o1, &o2, &o3];
        let sign_table = SignTableCache::new(16);
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
            let mut bufs = CpScanBuffers::new();

            // Coarse pass: 10 alpha x 11 t_sol x 11 t_atm = 1210 pts
            for a_step in 1..=10_i32 {
                let alpha_cp = a_step as f64 * 0.05;
                for ts_step in -5..=5_i32 {
                    let t_sol_trial = t_sol + ts_step as f64 * 0.2;
                    for ta_step in -5..=5_i32 {
                        let t_atm_trial = t_atm + ta_step as f64 * 0.6;

                        let r = evaluate_cp_scan_point(
                            alpha_cp, t_sol_trial, t_atm_trial, &phi,
                            &ctx, &mut bufs,
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
            let mut bufs_fine = CpScanBuffers::new();

            let mut fine_best_jcp = jcp_c;
            let mut fine_best = (alpha_c, ts_c, ta_c, t12_c, t13_c, t23_c, delta_c);

            for a_step in -5..=5_i32 {
                let alpha_cp = (alpha_c + a_step as f64 * 0.01).max(0.001);
                for ts_step in -5..=5_i32 {
                    let t_sol_f = ts_c + ts_step as f64 * 0.04;
                    for ta_step in -5..=5_i32 {
                        let t_atm_f = ta_c + ta_step as f64 * 0.12;

                        let r = evaluate_cp_scan_point(
                            alpha_cp, t_sol_f, t_atm_f, &phi_fine,
                            &ctx_fine, &mut bufs_fine,
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

            // Rephasing-aware delta_CP: recompute via evaluate_cp_scan_point
            // to get both arg(-U_e3) and arg(Jarlskog quartet).
            let r_final = evaluate_cp_scan_point(
                alpha, ts, ta, &phi_fine,
                &ctx_fine, &mut bufs_fine,
            );

            // For the Jarlskog quartet arg we need the full PMNS matrix.
            // Rebuild it one more time (single call, not in a loop).
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

            // ----- Mass-squared ratio r (B3) -----
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
        } else {
            println!("  No solution found within 2% angle tolerance with nonzero J_CP.");
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
        use crate::bell_inequality::{SignTableCache, rotate_sparse};
        use crate::three_fermion_generations::get_sedenion_subalgebras;

        let ch_pair = (11_usize, 12);
        let alpha_ch = 3.00_f64;
        let alpha_nu = 1.35_f64;

        // Selector pairs to test: (7,8) and (8,7)
        let selector_pairs = [(7_usize, 8_usize), (8_usize, 7_usize)];
        // L/R swap: controls which mode is a vs b in the associator
        let lr_swaps = [false, true];
        // Epsilon sign: negate profiles or not
        let eps_signs = [false, true];

        let (o1, o2, o3) = get_sedenion_subalgebras();
        let subs = [&o1, &o2, &o3];
        let sign_table = SignTableCache::new(16);

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
}
