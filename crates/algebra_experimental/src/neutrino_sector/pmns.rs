// Pdg2024 used in chi_squared_pmns; imported directly to avoid super:: chains.
use flavor_lifts::{Pdg2024, extract_pmns_angles};

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
    use crate::{
        bell_inequality::SignTableCache, lepton_mass_hierarchy::cd_braid_signed_friction,
        majorana_braiding::MajoranaMode, quark_sector::SubalgebraScheme,
        three_fermion_generations::get_sedenion_subalgebras,
    };

    // Casimir baseline via neutral projections + lepton assembly
    let cb = construct_casimir_baseline(SubalgebraScheme::InterleavedStride);
    let (m_baseline_ch, m_baseline_nu) = assemble_lepton_baseline(&cb);

    // Signed friction
    let (o1, o2, o3) = get_sedenion_subalgebras();
    let subs = [&o1, &o2, &o3];
    let sign_table = SignTableCache::new(16);

    let w1: f64 = -0.656850;
    let w2: f64 = -0.741999;

    let ch_a = MajoranaMode {
        gamma_index: charged_pair.0.saturating_sub(1),
        cd_basis_index: charged_pair.0,
        cd_dim: 16,
    };
    let ch_b = MajoranaMode {
        gamma_index: charged_pair.1.saturating_sub(1),
        cd_basis_index: charged_pair.1,
        cd_dim: 16,
    };
    let nu_a = MajoranaMode {
        gamma_index: neutrino_pair.0.saturating_sub(1),
        cd_basis_index: neutrino_pair.0,
        cd_dim: 16,
    };
    let nu_b = MajoranaMode {
        gamma_index: neutrino_pair.1.saturating_sub(1),
        cd_basis_index: neutrino_pair.1,
        cd_dim: 16,
    };

    let sel_ch: Vec<f64> = subs
        .iter()
        .map(|s| cd_braid_signed_friction(&ch_a, &ch_b, s, &sign_table))
        .collect();
    let sel_nu: Vec<f64> = subs
        .iter()
        .map(|s| cd_braid_signed_friction(&nu_a, &nu_b, s, &sign_table))
        .collect();

    // Baseline + cross-coupled friction perturbation
    let mut m_charged = m_baseline_ch;
    let mut m_neutrino = m_baseline_nu;
    for i in 0..3 {
        let f_ch = w1 * sel_ch[i] + w2 * sel_nu[i];
        let f_nu = w1 * sel_nu[i] + w2 * sel_ch[i];
        m_charged[(i, i)] += f_ch.exp();
        m_neutrino[(i, i)] += f_nu.exp();
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
    use crate::{
        bell_inequality::SignTableCache, lepton_mass_hierarchy::cd_braid_signed_friction,
        majorana_braiding::MajoranaMode, quark_sector::SubalgebraScheme,
        three_fermion_generations::get_sedenion_subalgebras,
    };
    use cd_kernel::gourlay_psi;

    // Casimir baseline via neutral projections + lepton assembly
    let cb = construct_casimir_baseline(SubalgebraScheme::InterleavedStride);
    let (m_base_ch, m_base_nu) = assemble_lepton_baseline(&cb);

    let (o1, o2, o3) = get_sedenion_subalgebras();
    let subs = [&o1, &o2, &o3];
    let sign_table = SignTableCache::new(16);

    let w1: f64 = -0.656850;
    let w2: f64 = -0.741999;

    let ch_a = MajoranaMode {
        gamma_index: charged_pair.0 - 1,
        cd_basis_index: charged_pair.0,
        cd_dim: 16,
    };
    let ch_b = MajoranaMode {
        gamma_index: charged_pair.1 - 1,
        cd_basis_index: charged_pair.1,
        cd_dim: 16,
    };
    let nu_a = MajoranaMode {
        gamma_index: neutrino_pair.0 - 1,
        cd_basis_index: neutrino_pair.0,
        cd_dim: 16,
    };
    let nu_b = MajoranaMode {
        gamma_index: neutrino_pair.1 - 1,
        cd_basis_index: neutrino_pair.1,
        cd_dim: 16,
    };

    let sel_ch: Vec<f64> = subs
        .iter()
        .map(|s| cd_braid_signed_friction(&ch_a, &ch_b, s, &sign_table))
        .collect();
    let sel_nu: Vec<f64> = subs
        .iter()
        .map(|s| cd_braid_signed_friction(&nu_a, &nu_b, s, &sign_table))
        .collect();

    // Build 3x3 friction tensors with off-diagonal terms
    let mut m_ch = m_base_ch;
    let mut m_nu = m_base_nu;

    // Diagonal terms (same as before)
    for i in 0..3 {
        let f_ch = w1 * sel_ch[i] + w2 * sel_nu[i];
        let f_nu = w1 * sel_nu[i] + w2 * sel_ch[i];
        m_ch[(i, i)] += f_ch.exp();
        m_nu[(i, i)] += f_nu.exp();
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
                if i == j {
                    continue;
                }
                let shift = (j + 3 - i) % 3;
                m_ch[(i, j)] += alpha_cross * circulant_ch[shift];
                m_nu[(i, j)] += alpha_cross * circulant_nu[shift];
            }
        }

        // Symmetrize
        for i in 0..3 {
            for j in (i + 1)..3 {
                let avg_ch = (m_ch[(i, j)] + m_ch[(j, i)]) / 2.0;
                let avg_nu = (m_nu[(i, j)] + m_nu[(j, i)]) / 2.0;
                m_ch[(i, j)] = avg_ch;
                m_ch[(j, i)] = avg_ch;
                m_nu[(i, j)] = avg_nu;
                m_nu[(j, i)] = avg_nu;
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
    let prod = u[(0, 1)] * u[(1, 2)] * u[(0, 2)] * u[(1, 1)];
    // For a real orthogonal matrix, the "imaginary part" is always zero.
    // We return the antisymmetric combination as a consistency check.
    let j = u[(0, 1)] * u[(1, 2)] * u[(2, 0)] - u[(0, 2)] * u[(1, 1)] * u[(2, 0)];
    // This is actually Re(U_e2 * U_mu3 * U_tau1) - Re(U_e3 * U_mu2 * U_tau1),
    // which is an antisymmetric product, NOT the Jarlskog invariant.
    // For a truly real orthogonal matrix, J = 0 by definition.
    let _ = prod;
    let _ = j;
    0.0
}

// extract_pmns_angles and Pdg2024 are re-exported from flavor_lifts::angles.
// The definitions below were byte-for-byte duplicates (CPD: 299 tokens, 48 lines).

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
pub fn compute_pmns(charged_pair: (usize, usize), neutrino_pair: (usize, usize)) -> PmnsResult {
    use faer::Side;

    let (m_ch, m_nu) = construct_pmns_matrices(charged_pair, neutrino_pair);

    let m_ch_sym = (&m_ch + m_ch.transpose()) * faer::Scale(0.5);
    let m_nu_sym = (&m_nu + m_nu.transpose()) * faer::Scale(0.5);

    let eig_ch = m_ch_sym.self_adjoint_eigen(Side::Lower).unwrap();
    let eig_nu = m_nu_sym.self_adjoint_eigen(Side::Lower).unwrap();

    // Sort eigenvectors by ascending absolute mass before constructing U_PMNS.
    // Required because faer guarantees ascending signed-eigenvalue order, but
    // generation ordering (nu_1 lightest, nu_3 heaviest) requires abs ordering.
    let (ch_masses, u_ch) =
        crate::quark_sector::sort_mass_eigenstates(&eig_ch.S(), &eig_ch.U());
    let (nu_masses, u_nu) =
        crate::quark_sector::sort_mass_eigenstates(&eig_nu.S(), &eig_nu.U());

    // U_PMNS = U_charged^T * U_neutrino
    let u_pmns_raw = u_ch.transpose() * u_nu;

    // Align columns to PDG convention: descending |Ue| content.
    let aligned = crate::quark_sector::align_pmns_columns(&u_pmns_raw);

    let (theta_12, theta_13, theta_23) = extract_pmns_angles(aligned.matrix());

    // Mass-squared differences (in arbitrary units, ratios are meaningful)
    let delta_m21_sq = nu_masses[1].powi(2) - nu_masses[0].powi(2);
    let delta_m31_sq = nu_masses[2].powi(2) - nu_masses[0].powi(2);

    let j = jarlskog_from_real_pmns(aligned.matrix());
    let cp_phase = extract_cp_phase((theta_12, theta_13, theta_23), j);

    PmnsResult {
        matrix: aligned.into_matrix(),
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

    crate::quark_sector::construct_casimir_projections(&basis, &complex_structure, scheme)
}

/// Assemble lepton baseline mass matrices from raw Casimir projections.
///
/// Currently uses the same convention as the quark sector (M_ch = C_SU3 + C_SU2,
/// M_nu = C_SU3 - C_SU2) to preserve regression. This is an explicit choice
/// that can be revisited independently of the quark sector.
pub(crate) fn assemble_lepton_baseline(
    cb: &crate::quark_sector::CasimirBaseline,
) -> (faer::Mat<f64>, faer::Mat<f64>) {
    crate::quark_sector::assemble_quark_matrices(cb)
}

/// Construct PMNS matrices with two independent psi-coupling parameters.
///
/// Factored from the `test_pmns_offdiag_two_param` scan body into a pure,
/// deterministic function. The construction is:
///
/// Shared initialization for psi-coupling PMNS functions.
///
/// Computes the Casimir baseline, signed-friction profiles per generation,
/// and applies the diagonal friction terms.  The returned matrices are ready
/// for the caller's off-diagonal psi-circulant coupling step.
///
/// Used by: `construct_pmns_matrices_two_param`, `construct_pmns_matrices_v6_modulated`.
// Private helper: (m_ch, m_nu, ch_profiles, nu_profiles).
// Returning four values as a tuple is intentional for a private one-caller helper.
#[allow(clippy::type_complexity)]
fn build_friction_matrices(
    charged_pair: (usize, usize),
    neutrino_pair: (usize, usize),
) -> (
    faer::Mat<f64>,
    faer::Mat<f64>,
    Vec<[f64; 16]>,
    Vec<[f64; 16]>,
) {
    use crate::{
        bell_inequality::{SignTableCache, rotate_sparse},
        lepton_mass_hierarchy::cd_braid_signed_friction,
        majorana_braiding::MajoranaMode,
        quark_sector::SubalgebraScheme,
        three_fermion_generations::get_sedenion_subalgebras,
    };

    let cb = construct_casimir_baseline(SubalgebraScheme::InterleavedStride);
    let (m_base_ch, m_base_nu) = assemble_lepton_baseline(&cb);

    let (o1, o2, o3) = get_sedenion_subalgebras();
    let subs = [&o1, &o2, &o3];
    let sign_table = SignTableCache::new(16);

    let w1: f64 = -0.656850;
    let w2: f64 = -0.741999;

    let ch_a = MajoranaMode {
        gamma_index: charged_pair.0 - 1,
        cd_basis_index: charged_pair.0,
        cd_dim: 16,
    };
    let ch_b = MajoranaMode {
        gamma_index: charged_pair.1 - 1,
        cd_basis_index: charged_pair.1,
        cd_dim: 16,
    };
    let nu_a = MajoranaMode {
        gamma_index: neutrino_pair.0 - 1,
        cd_basis_index: neutrino_pair.0,
        cd_dim: 16,
    };
    let nu_b = MajoranaMode {
        gamma_index: neutrino_pair.1 - 1,
        cd_basis_index: neutrino_pair.1,
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
            for &k in sub {
                if k == 0 || k == i || k == j {
                    continue;
                }
                let x_sparse = [(k, 1.0)];
                profile[k] = sign_table.sparse_associator_sum(&a_rotated, &x_sparse, &b_sparse);
            }
            profile
        };

    let ch_profiles: Vec<[f64; 16]> = subs
        .iter()
        .map(|s| build_profile(&ch_a, &ch_b, s))
        .collect();
    let nu_profiles: Vec<[f64; 16]> = subs
        .iter()
        .map(|s| build_profile(&nu_a, &nu_b, s))
        .collect();

    let sel_ch: Vec<f64> = subs
        .iter()
        .map(|s| cd_braid_signed_friction(&ch_a, &ch_b, s, &sign_table))
        .collect();
    let sel_nu: Vec<f64> = subs
        .iter()
        .map(|s| cd_braid_signed_friction(&nu_a, &nu_b, s, &sign_table))
        .collect();

    let mut m_ch = m_base_ch;
    let mut m_nu = m_base_nu;
    for i in 0..3 {
        let f_ch = w1 * sel_ch[i] + w2 * sel_nu[i];
        let f_nu = w1 * sel_nu[i] + w2 * sel_ch[i];
        m_ch[(i, i)] += f_ch.exp();
        m_nu[(i, i)] += f_nu.exp();
    }

    (m_ch, m_nu, ch_profiles, nu_profiles)
}

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
    use cd_kernel::gourlay_psi;

    let (mut m_ch, mut m_nu, ch_profiles, nu_profiles) =
        build_friction_matrices(charged_pair, neutrino_pair);

    let dot16 =
        |a: &[f64; 16], b: &[f64; 16]| -> f64 { a.iter().zip(b.iter()).map(|(x, y)| x * y).sum() };

    // Off-diagonal psi circulant coupling
    for i in 0..3 {
        for j in 0..3 {
            if i == j {
                continue;
            }
            let psi_nu_j = gourlay_psi(&nu_profiles[j]);
            let psi_ch_j = gourlay_psi(&ch_profiles[j]);
            m_nu[(i, j)] += alpha_nu * dot16(&nu_profiles[i], &psi_nu_j);
            m_ch[(i, j)] += alpha_ch * dot16(&ch_profiles[i], &psi_ch_j);
        }
    }

    // Step 4: Symmetrize
    for i in 0..3 {
        for j in (i + 1)..3 {
            let avg_ch = (m_ch[(i, j)] + m_ch[(j, i)]) / 2.0;
            let avg_nu = (m_nu[(i, j)] + m_nu[(j, i)]) / 2.0;
            m_ch[(i, j)] = avg_ch;
            m_ch[(j, i)] = avg_ch;
            m_nu[(i, j)] = avg_nu;
            m_nu[(j, i)] = avg_nu;
        }
    }

    let m_ch_s = (&m_ch + m_ch.transpose()) * faer::Scale(0.5);
    let m_nu_s = (&m_nu + m_nu.transpose()) * faer::Scale(0.5);

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
    use cd_kernel::gourlay_psi;

    let (mut m_ch, mut m_nu, ch_profiles, nu_profiles) =
        build_friction_matrices(charged_pair, neutrino_pair);

    let dot16 =
        |a: &[f64; 16], b: &[f64; 16]| -> f64 { a.iter().zip(b.iter()).map(|(x, y)| x * y).sum() };

    // Compute V_6 modulation field: collapse beta into 3 generation factors
    let n_basis = v6_basis.nrows().min(6);
    let n_cols = v6_basis.ncols().min(42);
    let mut v_combined = vec![0.0_f64; n_cols];
    for k in 0..n_basis {
        if beta[k].abs() < 1e-15 {
            continue;
        }
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

    // Off-diagonal psi coupling with V_6-modulated alpha
    for i in 0..3 {
        for j in 0..3 {
            if i == j {
                continue;
            }
            let psi_nu_j = gourlay_psi(&nu_profiles[j]);
            let psi_ch_j = gourlay_psi(&ch_profiles[j]);

            // Generation-pair-specific modulation
            let alpha_nu_ij = base_alpha_nu * (phi[i] + phi[j]).exp();
            let alpha_ch_ij = base_alpha_ch * (phi[i] + phi[j]).exp();

            m_nu[(i, j)] += alpha_nu_ij * dot16(&nu_profiles[i], &psi_nu_j);
            m_ch[(i, j)] += alpha_ch_ij * dot16(&ch_profiles[i], &psi_ch_j);
        }
    }

    // Symmetrize
    for i in 0..3 {
        for j in (i + 1)..3 {
            let avg_ch = (m_ch[(i, j)] + m_ch[(j, i)]) / 2.0;
            let avg_nu = (m_nu[(i, j)] + m_nu[(j, i)]) / 2.0;
            m_ch[(i, j)] = avg_ch;
            m_ch[(j, i)] = avg_ch;
            m_nu[(i, j)] = avg_nu;
            m_nu[(j, i)] = avg_nu;
        }
    }

    let m_ch_s = (&m_ch + m_ch.transpose()) * faer::Scale(0.5);
    let m_nu_s = (&m_nu + m_nu.transpose()) * faer::Scale(0.5);

    (m_ch_s, m_nu_s)
}
