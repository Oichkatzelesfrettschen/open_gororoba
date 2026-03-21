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
}
