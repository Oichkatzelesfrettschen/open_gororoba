//! The Quark Sector from the Sedenion Algebra
//!
//! Derives quark mass hierarchy and CKM mixing matrix from the sedenion SU(5)
//! structure. Supports two coexisting subalgebra definitions (contiguous-block
//! and interleaved-stride) and compares both against PDG measurements.
//!
//! # Architecture
//!
//! 1. **Ladder operators**: SU(3) Gell-Mann raising/lowering from StandardModelMapping
//! 2. **Mass matrices**: Casimir projected onto each subalgebra -> 3x3 M_up, M_down
//! 3. **CKM matrix**: V_CKM = U_up^T * U_down from diagonalization eigenvectors
//! 4. **Scheme comparison**: both subalgebra definitions evaluated and ranked vs PDG

use crate::{
    cayley_dickson_structs::Sedenion,
    neutrino_sector::{GeneratorType, classify_generator},
    quantum_state::QuantumState,
    sedenion_subalgebras::get_octonion_subalgebras,
    su_n_generators::construct_su5_generators_algebraic,
    three_fermion_generations::get_sedenion_subalgebras,
};
use faer::{Mat, Side};

/// Configuration for which subalgebra definition to use.
///
/// Three distinct schemes appear in the literature. They are NOT interchangeable:
///
/// - **TangContiguous** (Tang & Tang 2024): O1={e_0..e_7}, O2={e_0..e_3,e_8..e_11},
///   O3={e_0..e_3,e_12..e_15}. Shared quaternion = {e_0,e_1,e_2,e_3} (spacetime Gamma).
///   Mass-energy formulas and creation/annihilation operators written in this basis.
///
/// - **InterleavedStride** (de Marrais 2007, Gillard & Gresnigt 2019):
///   O1={0,1,4,5,8,9,12,13}, O2={0,2,4,6,8,10,12,14}, O3={0,3,4,7,8,11,12,15}.
///   Related by cyclic permutation sigma: k -> k+1 on generation-specific indices.
///   Common intersection = {e_0, e_4, e_8, e_12} (Theta quaternion).
///
/// - **GresnigtIntersecting** (Gresnigt 2019, arXiv:1904.03186):
///   Same index sets as InterleavedStride, but the S3 family symmetry is
///   implemented via Cl(8) automorphisms (psi_3, epsilon), NOT cyclic permutation
///   of basis indices. Gauge generators required to commute with S3 action.
///   Common quaternionic subalgebra = {1, e_1, e_14, e_15} in Gresnigt's basis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubalgebraScheme {
    /// Tang's contiguous CD-doubling blocks (U/V/W type).
    /// Shared quaternion = {e_0, e_1, e_2, e_3}.
    ContiguousBlock,
    /// Interleaved stride with cyclic S3 structure.
    /// Shared quaternion = {e_0, e_4, e_8, e_12} (Theta).
    InterleavedStride,
    /// Gresnigt's Cl(8) intersecting scheme.
    /// Uses same index sets as InterleavedStride but S3 action is via
    /// Cl(8) automorphisms, not basis permutation.
    GresnigtIntersecting,
}

/// Get the three octonionic subalgebra index sets for a given scheme.
pub fn get_subalgebras(scheme: SubalgebraScheme) -> [Vec<usize>; 3] {
    match scheme {
        SubalgebraScheme::ContiguousBlock => {
            let (o1, o2, o3) = get_octonion_subalgebras();
            [o1, o2, o3]
        }
        SubalgebraScheme::InterleavedStride | SubalgebraScheme::GresnigtIntersecting => {
            // Same index sets; the difference is the S3 action mechanism
            let (o1, o2, o3) = get_sedenion_subalgebras();
            [o1, o2, o3]
        }
    }
}

/// Raw Casimir projection matrices, before any sector-specific assembly.
///
/// Holds the 3x3 Gram matrices for SU(3) and SU(2) Casimir operators
/// projected onto the three octonionic subalgebras:
///   `c_su3[i,j] = Re(C_SU3_i^* C_SU3_j)`
///   `c_su2[i,j] = Re(C_SU2_i^* C_SU2_j)`
///
/// Sector-specific conventions (e.g., M_up = c_su3 + c_su2 for quarks)
/// belong in the assembler, not here.
pub struct CasimirBaseline {
    pub c_su3: Mat<f64>,
    pub c_su2: Mat<f64>,
}

/// Compute raw Casimir projection matrices for SU(3) and SU(2).
///
/// Pure function: extracts the SU(5) generators, computes the quadratic
/// Casimir for SU(3) and SU(2), projects each onto the three octonionic
/// subalgebras of the given scheme, and returns the two 3x3 Gram matrices.
///
/// The caller decides how to combine them (quark: +/-, lepton: any convention).
pub fn construct_casimir_projections(
    basis: &[Sedenion; 16],
    complex_structure: &Sedenion,
    scheme: SubalgebraScheme,
) -> CasimirBaseline {
    let su5_gens = construct_su5_generators_algebraic(basis, complex_structure);
    let subalgebras = get_subalgebras(scheme);

    // SU(3) Casimir: C_2 = sum_{a=0}^{7} T_a * T_a
    let su3_gens: Vec<QuantumState> = su5_gens
        .iter()
        .enumerate()
        .filter(|(i, g)| {
            classify_generator(*i) == GeneratorType::SU3 && **g != QuantumState::TopologicalNull
        })
        .map(|(_, g)| *g)
        .collect();

    let casimir_su3 = su3_gens
        .iter()
        .fold(QuantumState::Observable(Sedenion::default()), |acc, g| {
            acc + *g * *g
        });

    let casimir_s = match casimir_su3 {
        QuantumState::Observable(s) => s,
        QuantumState::TopologicalNull => Sedenion::default(),
    };

    // SU(2) Casimir
    let su2_gens: Vec<QuantumState> = su5_gens
        .iter()
        .enumerate()
        .filter(|(i, g)| {
            classify_generator(*i) == GeneratorType::SU2 && **g != QuantumState::TopologicalNull
        })
        .map(|(_, g)| *g)
        .collect();

    let casimir_su2 = su2_gens
        .iter()
        .fold(QuantumState::Observable(Sedenion::default()), |acc, g| {
            acc + *g * *g
        });

    let casimir_su2_s = match casimir_su2 {
        QuantumState::Observable(s) => s,
        QuantumState::TopologicalNull => Sedenion::default(),
    };

    // Project Casimir onto each subalgebra
    let proj_su3: Vec<Sedenion> = subalgebras
        .iter()
        .map(|sub| casimir_s.project_to_subalgebra(sub))
        .collect();

    let proj_su2: Vec<Sedenion> = subalgebras
        .iter()
        .map(|sub| casimir_su2_s.project_to_subalgebra(sub))
        .collect();

    // Build 3x3 Gram matrices (raw projections, no +/- combination)
    let mut c_su3 = Mat::<f64>::zeros(3, 3);
    let mut c_su2 = Mat::<f64>::zeros(3, 3);

    for i in 0..3 {
        for j in 0..3 {
            c_su3[(i, j)] = (proj_su3[i].conj() * proj_su3[j]).to_slice()[0];
            c_su2[(i, j)] = (proj_su2[i].conj() * proj_su2[j]).to_slice()[0];
        }
    }

    CasimirBaseline { c_su3, c_su2 }
}

/// Construct quark ladder operators as color triplets.
///
/// Uses the SU(3) generators (axes 1-8 from classify_generator) to build
/// Gell-Mann raising/lowering operators. Three generations from the three
/// octonionic subalgebras.
///
/// Returns (up_type, down_type) where each is Vec of 9 QuantumStates
/// (3 generations x 3 colors).
pub fn construct_quark_ladder_operators(
    basis: &[Sedenion; 16],
    complex_structure: &Sedenion,
    scheme: SubalgebraScheme,
) -> (Vec<QuantumState>, Vec<QuantumState>) {
    let su5_gens = construct_su5_generators_algebraic(basis, complex_structure);
    let subalgebras = get_subalgebras(scheme);

    // Extract SU(3) generators (indices 0-7 by classify_generator)
    let su3_gens: Vec<QuantumState> = su5_gens
        .iter()
        .enumerate()
        .filter(|(i, g)| {
            classify_generator(*i) == GeneratorType::SU3 && **g != QuantumState::TopologicalNull
        })
        .map(|(_, g)| *g)
        .collect();

    // Extract SU(2) generators (indices 8-10)
    let su2_gens: Vec<QuantumState> = su5_gens
        .iter()
        .enumerate()
        .filter(|(i, g)| {
            classify_generator(*i) == GeneratorType::SU2 && **g != QuantumState::TopologicalNull
        })
        .map(|(_, g)| *g)
        .collect();

    // Gell-Mann raising/lowering operators from SU(3) generators:
    //   T+ = (lambda_1 + i*lambda_2)/2  (u-d color transition)
    //   V+ = (lambda_4 + i*lambda_5)/2  (u-s color transition)
    //   U+ = (lambda_6 + i*lambda_7)/2  (d-s color transition)
    // where "i" here is the sedenion complex structure, not sqrt(-1).
    let i_op = QuantumState::Observable(*complex_structure);

    let make_raising = |gen_re: QuantumState, gen_im: QuantumState| -> QuantumState {
        (gen_re + i_op * gen_im) * 0.5
    };

    // Build color triplet raising operators (if we have enough SU(3) generators)
    let color_ops: Vec<QuantumState> = if su3_gens.len() >= 7 {
        vec![
            make_raising(su3_gens[0], su3_gens[1]), // T+ (lambda_1, lambda_2)
            make_raising(su3_gens[3], su3_gens[4]), // V+ (lambda_4, lambda_5)
            make_raising(su3_gens[5], su3_gens[6]), // U+ (lambda_6, lambda_7)
        ]
    } else {
        // Fallback: use available generators
        su3_gens.iter().take(3).copied().collect()
    };

    let mut up_type = Vec::with_capacity(9);
    let mut down_type = Vec::with_capacity(9);

    for sub in &subalgebras {
        for color_op in &color_ops {
            // Project the color operator onto this generation's subalgebra
            let projected_color = match color_op {
                QuantumState::Observable(s) => {
                    QuantumState::Observable(s.project_to_subalgebra(sub))
                }
                QuantumState::TopologicalNull => QuantumState::TopologicalNull,
            };

            // Up-type: combine with SU(2)_L upper component
            let su2_up = if !su2_gens.is_empty() {
                match su2_gens[0] {
                    QuantumState::Observable(s) => {
                        QuantumState::Observable(s.project_to_subalgebra(sub))
                    }
                    QuantumState::TopologicalNull => QuantumState::TopologicalNull,
                }
            } else {
                projected_color
            };
            up_type.push(projected_color + su2_up);

            // Down-type: combine with SU(2)_L lower component
            let su2_down = if su2_gens.len() >= 2 {
                match su2_gens[1] {
                    QuantumState::Observable(s) => {
                        QuantumState::Observable(s.project_to_subalgebra(sub))
                    }
                    QuantumState::TopologicalNull => QuantumState::TopologicalNull,
                }
            } else {
                projected_color
            };
            down_type.push(projected_color + su2_down);
        }
    }

    (up_type, down_type)
}

/// Build quark mass matrices from SU(3) Casimir projected onto subalgebras.
///
/// M_{ij} = Re(<C_2|O_i> . <C_2|O_j>*) where C_2 is the SU(3) quadratic Casimir
/// and |O_k> denotes projection onto the k-th subalgebra.
///
/// Returns (M_up, M_down) as 3x3 `faer::Mat<f64>`.
pub fn construct_quark_mass_matrices(
    basis: &[Sedenion; 16],
    complex_structure: &Sedenion,
    scheme: SubalgebraScheme,
) -> (Mat<f64>, Mat<f64>) {
    let cb = construct_casimir_projections(basis, complex_structure, scheme);
    assemble_quark_matrices(&cb)
}

/// Assemble quark mass matrices from raw Casimir projections.
///
/// Quark convention: M_up = C_SU3 + C_SU2, M_down = C_SU3 - C_SU2.
/// This is where the sector-specific sign choice lives.
pub fn assemble_quark_matrices(cb: &CasimirBaseline) -> (Mat<f64>, Mat<f64>) {
    let mut m_up = Mat::<f64>::zeros(3, 3);
    let mut m_down = Mat::<f64>::zeros(3, 3);

    for i in 0..3 {
        for j in 0..3 {
            let su3 = cb.c_su3[(i, j)];
            let su2 = cb.c_su2[(i, j)];
            m_up[(i, j)] = su3 + su2;
            m_down[(i, j)] = su3 - su2;
        }
    }

    (m_up, m_down)
}

/// Build quark mass matrices using signed topological friction.
///
/// Replaces the flavor-blind Casimir projection with generation-dependent
/// signed friction from oriented braiding.  Uses DIFFERENT braid-axis pairs
/// for up-type vs down-type sectors to ensure [H_u, H_d] != 0.
///
/// Up-type: braid axes (e_1, e_4) -> frictions {0, 2.83, -8.49}
/// Down-type: braid axes (e_5, e_8) -> frictions {0, -8.49, 2.83} (permuted)
///
/// Mass matrix: M_f = M_f^(0) + diag(exp(alpha * |signed_friction_i|))
/// where M_f^(0) is the Casimir baseline and alpha controls hierarchy steepness.
pub fn construct_quark_mass_matrices_with_friction(
    basis: &[Sedenion; 16],
    complex_structure: &Sedenion,
    scheme: SubalgebraScheme,
    alpha: f64,
) -> (Mat<f64>, Mat<f64>) {
    use crate::{
        bell_inequality::SignTableCache, lepton_mass_hierarchy::cd_braid_signed_friction,
        majorana_braiding::MajoranaMode, three_fermion_generations::get_sedenion_subalgebras,
    };

    // Get the Casimir baseline (rank 1, flavor-blind)
    let (m_up_0, m_down_0) = construct_quark_mass_matrices(basis, complex_structure, scheme);

    // Compute signed frictions for each generation
    let (o1, o2, o3) = get_sedenion_subalgebras();
    let subalgebras = [o1, o2, o3];
    let sign_table = SignTableCache::new(16);

    // Up-type: braid axes (e_1, e_4)
    let up_mode_a = MajoranaMode {
        gamma_index: 0,
        cd_basis_index: 1,
        cd_dim: 16,
    };
    let up_mode_b = MajoranaMode {
        gamma_index: 3,
        cd_basis_index: 4,
        cd_dim: 16,
    };
    let up_frictions: Vec<f64> = subalgebras
        .iter()
        .map(|sub| cd_braid_signed_friction(&up_mode_a, &up_mode_b, sub, &sign_table))
        .collect();

    // Down-type: braid axes (e_5, e_8) -- DIFFERENT pair for sector asymmetry
    let down_mode_a = MajoranaMode {
        gamma_index: 4,
        cd_basis_index: 5,
        cd_dim: 16,
    };
    let down_mode_b = MajoranaMode {
        gamma_index: 7,
        cd_basis_index: 8,
        cd_dim: 16,
    };
    let down_frictions: Vec<f64> = subalgebras
        .iter()
        .map(|sub| cd_braid_signed_friction(&down_mode_a, &down_mode_b, sub, &sign_table))
        .collect();

    // Perturb: M_f = M_f^(0) + diag(exp(alpha * |friction_i|))
    let mut m_up = m_up_0;
    let mut m_down = m_down_0;
    for i in 0..3 {
        let up_pert = (alpha * up_frictions[i].abs()).exp();
        let down_pert = (alpha * down_frictions[i].abs()).exp();
        m_up[(i, i)] += up_pert;
        m_down[(i, i)] += down_pert;
    }

    (m_up, m_down)
}

/// Build quark mass matrices using the fitted weighted signed friction composite.
///
/// Uses the lepton-sector fitted weights (difference-normalized):
///   w1=-0.6569, w2=-0.7420 (w_sym ~ -1/sqrt(2))
///
/// Up-type:   F_up   = w1 * Sel(e_1,e_4) + w2 * Sel(e_2,e_4)
/// Down-type: F_down = w1 * Sel(e_1,e_4) + w2 * Sel(e_3,e_4)
///
/// The shared first selector aligns the sectors partially (theta_12 ~ Cabibbo),
/// while the different second selectors misalign them (nonzero theta_13, theta_23).
pub fn construct_quark_mass_matrices_weighted_friction(
    basis: &[Sedenion; 16],
    complex_structure: &Sedenion,
    scheme: SubalgebraScheme,
) -> (Mat<f64>, Mat<f64>) {
    use crate::{
        bell_inequality::SignTableCache, lepton_mass_hierarchy::cd_braid_signed_friction,
        majorana_braiding::MajoranaMode, three_fermion_generations::get_sedenion_subalgebras,
    };

    let (m_up_0, m_down_0) = construct_quark_mass_matrices(basis, complex_structure, scheme);
    let (o1, o2, o3) = get_sedenion_subalgebras();
    let subs = [&o1, &o2, &o3];
    let sign_table = SignTableCache::new(16);

    // Fitted weights from the lepton mass hierarchy (difference-normalized fit)
    // Old (buggy F_e=0 normalization): w1=-0.9488, w2=-0.9609
    // New (correct exp(F_g - F_e) ratios): w1=-0.6569, w2=-0.7420
    // w_sym = -0.6994 ~ -1/sqrt(2), |w_asym/w_sym| = 0.061
    let w1 = -0.656850;
    let w2 = -0.741999;

    // Shared selector: Sel(e_1, e_4)
    let sel_shared_a = MajoranaMode {
        gamma_index: 0,
        cd_basis_index: 1,
        cd_dim: 16,
    };
    let sel_shared_b = MajoranaMode {
        gamma_index: 3,
        cd_basis_index: 4,
        cd_dim: 16,
    };
    let sel_shared: Vec<f64> = subs
        .iter()
        .map(|s| cd_braid_signed_friction(&sel_shared_a, &sel_shared_b, s, &sign_table))
        .collect();

    // Up-type second selector: Sel(e_2, e_4)
    let sel_up_a = MajoranaMode {
        gamma_index: 1,
        cd_basis_index: 2,
        cd_dim: 16,
    };
    let sel_up: Vec<f64> = subs
        .iter()
        .map(|s| cd_braid_signed_friction(&sel_up_a, &sel_shared_b, s, &sign_table))
        .collect();

    // Down-type second selector: Sel(e_3, e_4) -- DIFFERENT from up
    let sel_down_a = MajoranaMode {
        gamma_index: 2,
        cd_basis_index: 3,
        cd_dim: 16,
    };
    let sel_down: Vec<f64> = subs
        .iter()
        .map(|s| cd_braid_signed_friction(&sel_down_a, &sel_shared_b, s, &sign_table))
        .collect();

    // Composite friction per generation
    let mut m_up = m_up_0;
    let mut m_down = m_down_0;
    for i in 0..3 {
        let f_up = w1 * sel_shared[i] + w2 * sel_up[i];
        let f_down = w1 * sel_shared[i] + w2 * sel_down[i];
        m_up[(i, i)] += f_up.exp();
        m_down[(i, i)] += f_down.exp();
    }

    (m_up, m_down)
}

/// Build quark mass matrices with arbitrary selector pairs for the CKM scan.
///
/// Each sector uses two selectors: sel_a (shared or distinct) and sel_b,
/// combined with the lepton-fitted weights.
pub fn construct_quark_mass_matrices_scan(
    basis: &[Sedenion; 16],
    complex_structure: &Sedenion,
    scheme: SubalgebraScheme,
    up_pair: (usize, usize),   // (basis_idx_a, basis_idx_b) for up-type
    down_pair: (usize, usize), // (basis_idx_a, basis_idx_b) for down-type
) -> (Mat<f64>, Mat<f64>) {
    use crate::{
        bell_inequality::SignTableCache, lepton_mass_hierarchy::cd_braid_signed_friction,
        majorana_braiding::MajoranaMode,
    };

    let (m_up_0, m_down_0) = construct_quark_mass_matrices(basis, complex_structure, scheme);
    // Use the scheme-appropriate subalgebras for friction computation
    let subalgebras = get_subalgebras(scheme);
    let subs: Vec<&Vec<usize>> = subalgebras.iter().collect();
    let sign_table = SignTableCache::new(16);

    // Corrected lepton-sector weights (difference-normalized)
    let w1 = -0.656850;
    let w2 = -0.741999;

    let up_a = MajoranaMode {
        gamma_index: up_pair.0.saturating_sub(1),
        cd_basis_index: up_pair.0,
        cd_dim: 16,
    };
    let up_b = MajoranaMode {
        gamma_index: up_pair.1.saturating_sub(1),
        cd_basis_index: up_pair.1,
        cd_dim: 16,
    };
    let down_a = MajoranaMode {
        gamma_index: down_pair.0.saturating_sub(1),
        cd_basis_index: down_pair.0,
        cd_dim: 16,
    };
    let down_b = MajoranaMode {
        gamma_index: down_pair.1.saturating_sub(1),
        cd_basis_index: down_pair.1,
        cd_dim: 16,
    };

    let sel_up: Vec<f64> = subs
        .iter()
        .map(|s| cd_braid_signed_friction(&up_a, &up_b, s, &sign_table))
        .collect();
    let sel_down: Vec<f64> = subs
        .iter()
        .map(|s| cd_braid_signed_friction(&down_a, &down_b, s, &sign_table))
        .collect();

    // Use w1 for each sector's own selector, w2 for the other sector's selector
    // This creates cross-coupling that drives off-diagonal CKM
    let mut m_up = m_up_0;
    let mut m_down = m_down_0;
    for i in 0..3 {
        let f_up = w1 * sel_up[i] + w2 * sel_down[i];
        let f_down = w1 * sel_down[i] + w2 * sel_up[i];
        m_up[(i, i)] += f_up.exp();
        m_down[(i, i)] += f_down.exp();
    }

    (m_up, m_down)
}

/// Permutation-aware CKM extraction.
///
/// The raw V_CKM may be a cyclic permutation of the physical CKM because
/// the up and down sectors label generations in different orders. This
/// function tries all 6 row permutations x 6 column permutations (36 total)
/// and returns the representative closest to diagonal (maximizing sum |V'_ii|).
pub fn extract_ckm_permutation_aware(v_raw: &Mat<f64>) -> (Mat<f64>, [usize; 3], [usize; 3]) {
    let perms: [[usize; 3]; 6] = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];

    let mut best_diag = -1.0_f64;
    let mut best_v = v_raw.clone();
    let mut best_pu = [0, 1, 2];
    let mut best_pd = [0, 1, 2];

    for pu in &perms {
        for pd in &perms {
            // V'[i][j] = V_raw[pu[i]][pd[j]]
            let mut diag_sum = 0.0_f64;
            for i in 0..3 {
                diag_sum += v_raw[(pu[i], pd[i])].abs();
            }
            if diag_sum > best_diag {
                best_diag = diag_sum;
                best_pu = *pu;
                best_pd = *pd;
                for (i, &pi) in pu.iter().enumerate() {
                    for (j, &pj) in pd.iter().enumerate() {
                        best_v[(i, j)] = v_raw[(pi, pj)];
                    }
                }
            }
        }
    }

    (best_v, best_pu, best_pd)
}

/// CKM matrix derivation result.
pub struct CkmResult {
    /// The 3x3 CKM matrix.
    pub matrix: Mat<f64>,
    /// Mixing angles in degrees: (theta_12, theta_13, theta_23).
    pub angles_deg: (f64, f64, f64),
    /// CP-violation phase in degrees.
    pub cp_phase_deg: f64,
    /// Up-type mass eigenvalues (sorted ascending).
    pub up_masses: [f64; 3],
    /// Down-type mass eigenvalues (sorted ascending).
    pub down_masses: [f64; 3],
}

/// Sort eigenvalues by absolute mass and reorder eigenvector columns.
///
/// Enforces the Generation 1 < 2 < 3 mass hierarchy before CKM/PMNS
/// extraction, eliminating 90-degree permutation artifacts from
/// eigendecomposition ordering ambiguity.
pub fn sort_mass_eigenstates(
    evals: &faer::diag::DiagRef<'_, f64>,
    evecs: &faer::MatRef<'_, f64>,
) -> ([f64; 3], Mat<f64>) {
    let mut pairs: [(usize, f64); 3] = [
        (0, evals.column_vector()[0].abs()),
        (1, evals.column_vector()[1].abs()),
        (2, evals.column_vector()[2].abs()),
    ];
    pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let mut sorted_masses = [0.0_f64; 3];
    let mut sorted_evecs = Mat::<f64>::zeros(3, 3);
    for (new_idx, &(old_idx, mass)) in pairs.iter().enumerate() {
        sorted_masses[new_idx] = mass;
        for row in 0..3 {
            sorted_evecs[(row, new_idx)] = evecs[(row, old_idx)];
        }
    }
    (sorted_masses, sorted_evecs)
}

/// A PMNS matrix that has been explicitly aligned to the PDG column convention.
///
/// The only two constructors are `align_pmns_columns` (automatic PDG sort) and
/// `AlignedPmns::from_permuted_raw` (caller-supplied permutation, for finite-difference
/// gradient contexts where column ordering must stay fixed across perturbations).
///
/// WHY a newtype instead of a bare `Mat<f64>`: the PDG angle extraction formula
/// reads specific matrix elements by position (e.g. U_e3 = u[(0,2)]). If the
/// columns are in the wrong order those positions mean different angles. The
/// compiler can now reject any unaligned matrix from flowing into the PMNS pipeline.
pub struct AlignedPmns {
    matrix: Mat<f64>,
    col_perm: [usize; 3],
}

impl AlignedPmns {
    /// Apply explicit row and column permutations to `u_raw`.
    ///
    /// Use this when a reference-point alignment permutation (`perm_col`) has
    /// already been computed via `align_pmns_columns` and must be applied
    /// consistently to a perturbed matrix (e.g. for finite-difference gradients).
    /// The caller is responsible for ensuring the supplied permutation produces
    /// a physically meaningful ordering for their context.
    pub fn from_permuted_raw(
        u_raw: &Mat<f64>,
        perm_row: &[usize; 3],
        perm_col: &[usize; 3],
    ) -> Self {
        let mut matrix = Mat::<f64>::zeros(3, 3);
        for (i, &row) in perm_row.iter().enumerate() {
            for (j, &col) in perm_col.iter().enumerate() {
                matrix[(i, j)] = u_raw[(row, col)];
            }
        }
        Self {
            matrix,
            col_perm: *perm_col,
        }
    }

    /// Borrow the underlying matrix.
    pub fn matrix(&self) -> &Mat<f64> {
        &self.matrix
    }

    /// Return the column permutation that was applied (index `j` holds the
    /// original column that ended up in position `j` of the aligned matrix).
    pub fn col_perm(&self) -> [usize; 3] {
        self.col_perm
    }

    /// Consume the newtype, returning the inner matrix.
    pub fn into_matrix(self) -> Mat<f64> {
        self.matrix
    }
}

/// Sort PMNS matrix columns to PDG convention: descending |first-row element|.
///
/// The PDG PMNS convention assigns nu_1 as the state with the largest |Ue|^2
/// (solar mixing partner), and nu_3 as the state with the smallest |Ue|^2
/// (reactor angle). Sorting columns by descending `|U[0,j]|` enforces this,
/// ensuring `theta_13 = asin(|U[0,2]|)` is the smallest angle.
///
/// Returns an `AlignedPmns` whose `.col_perm()` records which original column
/// ended up in each position.
pub fn align_pmns_columns(u: &Mat<f64>) -> AlignedPmns {
    let mut cols: [(usize, f64); 3] = [
        (0, u[(0, 0)].abs()),
        (1, u[(0, 1)].abs()),
        (2, u[(0, 2)].abs()),
    ];
    // Sort descending: column with largest |Ue| goes to position 0 (nu_1),
    // column with smallest |Ue| goes to position 2 (nu_3).
    cols.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut aligned = Mat::<f64>::zeros(3, 3);
    let mut perm = [0usize; 3];
    for (new_j, &(old_j, _)) in cols.iter().enumerate() {
        perm[new_j] = old_j;
        for i in 0..3 {
            aligned[(i, new_j)] = u[(i, old_j)];
        }
    }
    AlignedPmns {
        matrix: aligned,
        col_perm: perm,
    }
}

/// Derive CKM matrix: V_CKM = U_up^T * U_down.
pub fn derive_ckm_matrix(
    basis: &[Sedenion; 16],
    complex_structure: &Sedenion,
    scheme: SubalgebraScheme,
) -> CkmResult {
    let (m_up, m_down) = construct_quark_mass_matrices(basis, complex_structure, scheme);

    // Symmetrize (numerical safety)
    let m_up_sym = (&m_up + m_up.transpose()) * faer::Scale(0.5);
    let m_down_sym = (&m_down + m_down.transpose()) * faer::Scale(0.5);

    let eig_up = m_up_sym.self_adjoint_eigen(Side::Lower).unwrap();
    let eig_down = m_down_sym.self_adjoint_eigen(Side::Lower).unwrap();

    // Sort by ascending absolute mass BEFORE extracting CKM
    let (up_masses, u_up) = sort_mass_eigenstates(&eig_up.S(), &eig_up.U());
    let (down_masses, u_down) = sort_mass_eigenstates(&eig_down.S(), &eig_down.U());

    // V_CKM = U_up^T * U_down
    let v_ckm = u_up.transpose() * &u_down;

    // Extract angles from the standard parameterization:
    //   V_us = sin(theta_12)*cos(theta_13)
    //   V_ub = sin(theta_13)
    //   V_cb = sin(theta_23)*cos(theta_13)
    let v_us = v_ckm[(0, 1)].abs();
    let v_ub = v_ckm[(0, 2)].abs();
    let v_cb = v_ckm[(1, 2)].abs();
    let _v_ud = v_ckm[(0, 0)].abs();
    let _v_tb = v_ckm[(2, 2)].abs();

    let theta_13 = v_ub.asin();
    let cos_13 = theta_13.cos();
    let theta_12 = if cos_13 > 1e-15 {
        (v_us / cos_13).min(1.0).asin()
    } else {
        0.0
    };
    let theta_23 = if cos_13 > 1e-15 {
        (v_cb / cos_13).min(1.0).asin()
    } else {
        0.0
    };

    // Jarlskog invariant: J = Im(V_us * V_cb * V_ub* * V_cs*)
    // For real matrices, J = 0 (no CP violation at tree level)
    let v_cs = v_ckm[(1, 1)];
    let j_invariant = v_ckm[(0, 1)] * v_ckm[(1, 2)] * v_ckm[(0, 2)] * v_cs;
    let cp_phase_deg = if j_invariant.abs() > 1e-15 {
        j_invariant.asin().to_degrees()
    } else {
        0.0
    };

    // Masses already sorted ascending by sort_mass_eigenstates
    CkmResult {
        matrix: v_ckm,
        angles_deg: (
            theta_12.to_degrees(),
            theta_13.to_degrees(),
            theta_23.to_degrees(),
        ),
        cp_phase_deg,
        up_masses,
        down_masses,
    }
}

/// Quark mass ratios from a given subalgebra scheme.
pub struct QuarkMassRatios {
    pub m_u_over_m_d: f64,
    pub m_c_over_m_s: f64,
    pub m_t_over_m_b: f64,
    pub scheme: SubalgebraScheme,
}

/// Full comparison of both subalgebra schemes.
pub struct SchemeComparison {
    pub contiguous: QuarkMassRatios,
    pub interleaved: QuarkMassRatios,
    pub contiguous_ckm: CkmResult,
    pub interleaved_ckm: CkmResult,
    /// Which scheme better matches PDG mass ratios.
    pub closer_to_pdg: SubalgebraScheme,
}

/// Compute mass ratios from CKM result.
fn mass_ratios_from_ckm(ckm: &CkmResult, scheme: SubalgebraScheme) -> QuarkMassRatios {
    let up = &ckm.up_masses;
    let down = &ckm.down_masses;

    // Ratios: m_u/m_d, m_c/m_s, m_t/m_b (lightest/lightest, mid/mid, heavy/heavy)
    let m_u_over_m_d = if down[0] > 1e-15 {
        up[0] / down[0]
    } else {
        f64::INFINITY
    };
    let m_c_over_m_s = if down[1] > 1e-15 {
        up[1] / down[1]
    } else {
        f64::INFINITY
    };
    let m_t_over_m_b = if down[2] > 1e-15 {
        up[2] / down[2]
    } else {
        f64::INFINITY
    };

    QuarkMassRatios {
        m_u_over_m_d,
        m_c_over_m_s,
        m_t_over_m_b,
        scheme,
    }
}

/// PDG 2025 quark mass ratio targets.
///
/// Ratios are scheme- and scale-dependent; only ratios evaluated in a common
/// scheme/scale have clean cancellation properties (PDG Review 2025).
///
/// - m_u/m_d: 0.473 +/- 0.025 at mu = 2 GeV in MSbar
/// - m_c/m_s: 11.77 +/- 0.25 at mu = 2 GeV in MSbar
/// - m_t/m_b: ~41.3 (naive m_t^pole/m_b^MSbar; mixed scheme, use with caution)
///
/// CKM global fit (PDG 2025):
///   sin(theta_12) = 0.22501, sin(theta_23) = 0.04183, sin(theta_13) = 0.003732
///   delta_CP ~ 1.147 rad, J_Jarlskog ~ 3.1e-5
const PDG_MU_MD: f64 = 0.473;
const PDG_MC_MS: f64 = 11.77;
const PDG_MT_MB: f64 = 41.3;

// CKM target angles (degrees) for comparison
#[allow(dead_code)]
const PDG_CKM_THETA12_DEG: f64 = 12.99;
#[allow(dead_code)]
const PDG_CKM_THETA23_DEG: f64 = 2.40;
#[allow(dead_code)]
const PDG_CKM_THETA13_DEG: f64 = 0.214;
#[allow(dead_code)]
const PDG_CKM_JARLSKOG: f64 = 3.1e-5;

/// Distance metric from PDG targets (sum of log-ratio deviations).
fn pdg_distance(ratios: &QuarkMassRatios) -> f64 {
    let log_dev = |predicted: f64, target: f64| -> f64 {
        if predicted > 0.0 && predicted.is_finite() {
            (predicted.ln() - target.ln()).abs()
        } else {
            f64::INFINITY
        }
    };

    log_dev(ratios.m_u_over_m_d, PDG_MU_MD)
        + log_dev(ratios.m_c_over_m_s, PDG_MC_MS)
        + log_dev(ratios.m_t_over_m_b, PDG_MT_MB)
}

/// Run the full quark pipeline for both subalgebra schemes and compare.
pub fn compare_schemes(basis: &[Sedenion; 16], complex_structure: &Sedenion) -> SchemeComparison {
    let ckm_c = derive_ckm_matrix(basis, complex_structure, SubalgebraScheme::ContiguousBlock);
    let ckm_i = derive_ckm_matrix(
        basis,
        complex_structure,
        SubalgebraScheme::InterleavedStride,
    );

    let ratios_c = mass_ratios_from_ckm(&ckm_c, SubalgebraScheme::ContiguousBlock);
    let ratios_i = mass_ratios_from_ckm(&ckm_i, SubalgebraScheme::InterleavedStride);

    let dist_c = pdg_distance(&ratios_c);
    let dist_i = pdg_distance(&ratios_i);

    let closer = if dist_c <= dist_i {
        SubalgebraScheme::ContiguousBlock
    } else {
        SubalgebraScheme::InterleavedStride
    };

    SchemeComparison {
        contiguous: ratios_c,
        interleaved: ratios_i,
        contiguous_ckm: ckm_c,
        interleaved_ckm: ckm_i,
        closer_to_pdg: closer,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn standard_basis_and_cs() -> ([Sedenion; 16], Sedenion) {
        let mut basis = [Sedenion::default(); 16];
        for i in 0..16 {
            let mut components = [0.0; 16];
            components[i] = 1.0;
            basis[i] = Sedenion::from_slice(&components);
        }
        let cs = basis[15];
        (basis, cs)
    }

    #[test]
    fn test_quark_ladder_operators_nonempty() {
        let (basis, cs) = standard_basis_and_cs();

        for scheme in [
            SubalgebraScheme::ContiguousBlock,
            SubalgebraScheme::InterleavedStride,
        ] {
            let (up, down) = construct_quark_ladder_operators(&basis, &cs, scheme);

            println!("--- QUARK LADDER OPERATORS ({scheme:?}) ---");
            println!("  up-type count: {}", up.len());
            println!("  down-type count: {}", down.len());

            let up_non_null = up
                .iter()
                .filter(|q| **q != QuantumState::TopologicalNull)
                .count();
            let down_non_null = down
                .iter()
                .filter(|q| **q != QuantumState::TopologicalNull)
                .count();
            println!("  up-type non-null: {up_non_null}");
            println!("  down-type non-null: {down_non_null}");

            assert!(!up.is_empty(), "{scheme:?}: up-type must not be empty");
            assert!(!down.is_empty(), "{scheme:?}: down-type must not be empty");
        }
    }

    #[test]
    fn test_quark_mass_matrices_symmetric() {
        let (basis, cs) = standard_basis_and_cs();

        for scheme in [
            SubalgebraScheme::ContiguousBlock,
            SubalgebraScheme::InterleavedStride,
        ] {
            let (m_up, m_down) = construct_quark_mass_matrices(&basis, &cs, scheme);

            println!("--- QUARK MASS MATRICES ({scheme:?}) ---");
            println!("  M_up = {:?}", m_up);
            println!("  M_down = {:?}", m_down);

            // Check symmetry
            for i in 0..3 {
                for j in 0..3 {
                    assert!(
                        (m_up[(i, j)] - m_up[(j, i)]).abs() < 1e-12,
                        "{scheme:?}: M_up not symmetric at ({i},{j})"
                    );
                    assert!(
                        (m_down[(i, j)] - m_down[(j, i)]).abs() < 1e-12,
                        "{scheme:?}: M_down not symmetric at ({i},{j})"
                    );
                }
            }
        }
    }

    #[test]
    fn test_ckm_unitarity() {
        let (basis, cs) = standard_basis_and_cs();

        for scheme in [
            SubalgebraScheme::ContiguousBlock,
            SubalgebraScheme::InterleavedStride,
        ] {
            let ckm = derive_ckm_matrix(&basis, &cs, scheme);

            println!("--- CKM MATRIX ({scheme:?}) ---");
            println!("  V_CKM = {:?}", ckm.matrix);
            println!(
                "  theta_12={:.2} deg, theta_13={:.2} deg, theta_23={:.2} deg",
                ckm.angles_deg.0, ckm.angles_deg.1, ckm.angles_deg.2
            );
            println!("  CP phase = {:.2} deg", ckm.cp_phase_deg);
            println!("  up masses = {:?}", ckm.up_masses);
            println!("  down masses = {:?}", ckm.down_masses);

            // Check unitarity: V * V^T = I
            let vvt = &ckm.matrix * ckm.matrix.transpose();
            for i in 0..3 {
                for j in 0..3 {
                    let expected = if i == j { 1.0 } else { 0.0 };
                    assert!(
                        (vvt[(i, j)] - expected).abs() < 1e-10,
                        "{scheme:?}: V*V^T not identity at ({i},{j}): got {:.6e}",
                        vvt[(i, j)]
                    );
                }
            }

            // Angles must be in [0, pi/2]
            assert!(ckm.angles_deg.0 >= 0.0 && ckm.angles_deg.0 <= 90.0);
            assert!(ckm.angles_deg.1 >= 0.0 && ckm.angles_deg.1 <= 90.0);
            assert!(ckm.angles_deg.2 >= 0.0 && ckm.angles_deg.2 <= 90.0);
        }
    }

    #[test]
    fn test_scheme_comparison() {
        let (basis, cs) = standard_basis_and_cs();
        let comparison = compare_schemes(&basis, &cs);

        println!("--- SCHEME COMPARISON ---");
        println!("Contiguous-block:");
        println!(
            "  m_u/m_d = {:.4} (PDG: {PDG_MU_MD})",
            comparison.contiguous.m_u_over_m_d
        );
        println!(
            "  m_c/m_s = {:.4} (PDG: {PDG_MC_MS})",
            comparison.contiguous.m_c_over_m_s
        );
        println!(
            "  m_t/m_b = {:.4} (PDG: {PDG_MT_MB})",
            comparison.contiguous.m_t_over_m_b
        );

        println!("Interleaved-stride:");
        println!(
            "  m_u/m_d = {:.4} (PDG: {PDG_MU_MD})",
            comparison.interleaved.m_u_over_m_d
        );
        println!(
            "  m_c/m_s = {:.4} (PDG: {PDG_MC_MS})",
            comparison.interleaved.m_c_over_m_s
        );
        println!(
            "  m_t/m_b = {:.4} (PDG: {PDG_MT_MB})",
            comparison.interleaved.m_t_over_m_b
        );

        println!("Closer to PDG: {:?}", comparison.closer_to_pdg);

        // Both schemes must produce non-NaN ratios (infinity is acceptable
        // when the lightest eigenvalue is zero -- this is a physically meaningful
        // degenerate mass matrix from the algebraic structure).
        for ratios in [&comparison.contiguous, &comparison.interleaved] {
            assert!(
                !ratios.m_u_over_m_d.is_nan(),
                "{:?}: m_u/m_d is NaN",
                ratios.scheme
            );
            assert!(
                !ratios.m_c_over_m_s.is_nan(),
                "{:?}: m_c/m_s is NaN",
                ratios.scheme
            );
            assert!(
                !ratios.m_t_over_m_b.is_nan(),
                "{:?}: m_t/m_b is NaN",
                ratios.scheme
            );
        }
    }

    /// Diagnostic: analyze the up-type mass matrix zero mode and CKM alignment.
    ///
    /// Per the peer-review audit:
    /// 1. det(M_up): if exactly zero, there is a genuine residual symmetry.
    /// 2. Smallest singular value: confirms the zero mode to machine precision.
    /// 3. Null vector: reveals which generation combination is massless.
    /// 4. ||[H_u, H_d]||_F: CKM alignment bottleneck diagnostic.
    #[test]
    fn test_quark_mass_matrix_diagnostics() {
        let (basis, cs) = standard_basis_and_cs();

        for scheme in [
            SubalgebraScheme::ContiguousBlock,
            SubalgebraScheme::InterleavedStride,
        ] {
            let (m_up, m_down) = construct_quark_mass_matrices(&basis, &cs, scheme);
            let m_up_sym = (&m_up + m_up.transpose()) * faer::Scale(0.5);
            let m_down_sym = (&m_down + m_down.transpose()) * faer::Scale(0.5);

            let eig_up = m_up_sym.self_adjoint_eigen(Side::Lower).unwrap();
            let eig_down = m_down_sym.self_adjoint_eigen(Side::Lower).unwrap();
            let s_up = eig_up.S();
            let s_down = eig_down.S();

            let mut up_evals = [0.0_f64; 3];
            let mut down_evals = [0.0_f64; 3];
            for i in 0..3 {
                up_evals[i] = s_up.column_vector()[i];
                down_evals[i] = s_down.column_vector()[i];
            }

            let det_up: f64 = up_evals.iter().product();
            let det_down: f64 = down_evals.iter().product();
            let min_sv_up = up_evals
                .iter()
                .map(|x| x.abs())
                .fold(f64::INFINITY, f64::min);
            let min_sv_down = down_evals
                .iter()
                .map(|x| x.abs())
                .fold(f64::INFINITY, f64::min);

            println!("--- {:?} DIAGNOSTICS ---", scheme);
            println!("  M_up eigenvalues: {:?}", up_evals);
            println!("  M_down eigenvalues: {:?}", down_evals);
            println!("  det(M_up) = {:.6e}", det_up);
            println!("  det(M_down) = {:.6e}", det_down);
            println!("  min |lambda|(M_up) = {:.6e}", min_sv_up);
            println!("  min |lambda|(M_down) = {:.6e}", min_sv_down);

            // H_u = M_up * M_up^T, H_d = M_down * M_down^T
            let h_u = &m_up_sym * m_up_sym.transpose();
            let h_d = &m_down_sym * m_down_sym.transpose();

            // Commutator [H_u, H_d] Frobenius norm
            let comm = &h_u * &h_d - &h_d * &h_u;
            let mut comm_frob = 0.0_f64;
            for i in 0..3 {
                for j in 0..3 {
                    comm_frob += comm[(i, j)].powi(2);
                }
            }
            comm_frob = comm_frob.sqrt();
            println!("  ||[H_u, H_d]||_F = {:.6e}", comm_frob);
            println!("  (small => CKM near identity; need nonzero for mixing)");

            if min_sv_up < 1e-10 {
                println!("  ** UP-TYPE ZERO MODE CONFIRMED (structural symmetry) **");
            }
        }
    }

    /// Inject signed friction into quark mass matrices and check all diagnostics.
    #[test]
    fn test_quark_mass_with_signed_friction() {
        let (basis, cs) = standard_basis_and_cs();
        let alpha = 1.0; // Start with alpha=1.0 (exp(|friction|))

        let (m_up, m_down) = construct_quark_mass_matrices_with_friction(
            &basis,
            &cs,
            SubalgebraScheme::InterleavedStride,
            alpha,
        );

        let m_up_sym = (&m_up + m_up.transpose()) * faer::Scale(0.5);
        let m_down_sym = (&m_down + m_down.transpose()) * faer::Scale(0.5);

        let eig_up = m_up_sym.self_adjoint_eigen(Side::Lower).unwrap();
        let eig_down = m_down_sym.self_adjoint_eigen(Side::Lower).unwrap();

        let mut up_evals = [0.0_f64; 3];
        let mut down_evals = [0.0_f64; 3];
        for i in 0..3 {
            up_evals[i] = eig_up.S().column_vector()[i];
            down_evals[i] = eig_down.S().column_vector()[i];
        }

        let det_up: f64 = up_evals.iter().product();
        let det_down: f64 = down_evals.iter().product();

        println!("--- QUARK MASS WITH SIGNED FRICTION (alpha={}) ---", alpha);
        println!("  M_up eigenvalues: {:?}", up_evals);
        println!("  M_down eigenvalues: {:?}", down_evals);
        println!("  det(M_up) = {:.6e}", det_up);
        println!("  det(M_down) = {:.6e}", det_down);

        // Check rank: are all eigenvalues nonzero?
        let up_rank = up_evals.iter().filter(|x| x.abs() > 1e-6).count();
        let down_rank = down_evals.iter().filter(|x| x.abs() > 1e-6).count();
        println!("  rank(M_up) = {}, rank(M_down) = {}", up_rank, down_rank);

        // Commutator [H_u, H_d]
        let h_u = &m_up_sym * m_up_sym.transpose();
        let h_d = &m_down_sym * m_down_sym.transpose();
        let comm = &h_u * &h_d - &h_d * &h_u;
        let mut comm_frob = 0.0_f64;
        for i in 0..3 {
            for j in 0..3 {
                comm_frob += comm[(i, j)].powi(2);
            }
        }
        comm_frob = comm_frob.sqrt();
        println!("  ||[H_u, H_d]||_F = {:.6e}", comm_frob);

        // CKM
        let u_up = eig_up.U();
        let u_down = eig_down.U();
        let v_ckm = u_up.transpose() * u_down;
        let v_us = v_ckm[(0, 1)].abs();
        let v_ub = v_ckm[(0, 2)].abs();
        let v_cb = v_ckm[(1, 2)].abs();
        let theta_13 = v_ub.asin();
        let cos_13 = theta_13.cos();
        let theta_12 = if cos_13 > 1e-15 {
            (v_us / cos_13).min(1.0).asin()
        } else {
            0.0
        };
        let theta_23 = if cos_13 > 1e-15 {
            (v_cb / cos_13).min(1.0).asin()
        } else {
            0.0
        };

        println!(
            "  CKM angles: theta_12={:.2} deg, theta_13={:.4} deg, theta_23={:.2} deg",
            theta_12.to_degrees(),
            theta_13.to_degrees(),
            theta_23.to_degrees()
        );
        println!("  PDG target: theta_12=12.99, theta_13=0.214, theta_23=2.40");

        // Mass ratios
        up_evals.sort_by(|a, b| a.abs().partial_cmp(&b.abs()).unwrap());
        down_evals.sort_by(|a, b| a.abs().partial_cmp(&b.abs()).unwrap());
        if down_evals[0].abs() > 1e-10 {
            println!(
                "  m_u/m_d = {:.4} (PDG: {PDG_MU_MD})",
                up_evals[0].abs() / down_evals[0].abs()
            );
        }
        if down_evals[1].abs() > 1e-10 {
            println!(
                "  m_c/m_s = {:.4} (PDG: {PDG_MC_MS})",
                up_evals[1].abs() / down_evals[1].abs()
            );
        }
        if down_evals[2].abs() > 1e-10 {
            println!(
                "  m_t/m_b = {:.4} (PDG: {PDG_MT_MB})",
                up_evals[2].abs() / down_evals[2].abs()
            );
        }
    }

    /// Full quark sector with weighted signed friction composite.
    #[test]
    fn test_quark_mass_weighted_composite() {
        let (basis, cs) = standard_basis_and_cs();
        let (m_up, m_down) = construct_quark_mass_matrices_weighted_friction(
            &basis,
            &cs,
            SubalgebraScheme::InterleavedStride,
        );

        let m_up_sym = (&m_up + m_up.transpose()) * faer::Scale(0.5);
        let m_down_sym = (&m_down + m_down.transpose()) * faer::Scale(0.5);

        let eig_up = m_up_sym.self_adjoint_eigen(Side::Lower).unwrap();
        let eig_down = m_down_sym.self_adjoint_eigen(Side::Lower).unwrap();

        let mut up_evals = [0.0_f64; 3];
        let mut down_evals = [0.0_f64; 3];
        for i in 0..3 {
            up_evals[i] = eig_up.S().column_vector()[i];
            down_evals[i] = eig_down.S().column_vector()[i];
        }

        println!("--- QUARK MASS WITH WEIGHTED COMPOSITE ---");
        println!("  M_up eigenvalues: {:?}", up_evals);
        println!("  M_down eigenvalues: {:?}", down_evals);

        let up_rank = up_evals.iter().filter(|x| x.abs() > 1e-6).count();
        let down_rank = down_evals.iter().filter(|x| x.abs() > 1e-6).count();
        println!("  rank(M_up) = {}, rank(M_down) = {}", up_rank, down_rank);

        // Commutator
        let h_u = &m_up_sym * m_up_sym.transpose();
        let h_d = &m_down_sym * m_down_sym.transpose();
        let comm = &h_u * &h_d - &h_d * &h_u;
        let mut comm_frob = 0.0_f64;
        for i in 0..3 {
            for j in 0..3 {
                comm_frob += comm[(i, j)].powi(2);
            }
        }
        comm_frob = comm_frob.sqrt();
        println!("  ||[H_u, H_d]||_F = {:.6e}", comm_frob);

        // CKM with permutation-aware extraction
        let u_up = eig_up.U();
        let u_down = eig_down.U();
        let v_raw = u_up.transpose() * u_down;

        println!("  V_CKM (raw):");
        for r in 0..3 {
            println!(
                "    [{:.6}, {:.6}, {:.6}]",
                v_raw[(r, 0)],
                v_raw[(r, 1)],
                v_raw[(r, 2)]
            );
        }

        let (v_ckm, pu, pd) = extract_ckm_permutation_aware(&v_raw);
        println!("  Permutation: up={:?}, down={:?}", pu, pd);
        println!("  V_CKM (aligned):");
        for r in 0..3 {
            println!(
                "    [{:.6}, {:.6}, {:.6}]",
                v_ckm[(r, 0)],
                v_ckm[(r, 1)],
                v_ckm[(r, 2)]
            );
        }

        let v_us = v_ckm[(0, 1)].abs();
        let v_ub = v_ckm[(0, 2)].abs();
        let v_cb = v_ckm[(1, 2)].abs();
        let theta_13 = v_ub.asin();
        let cos_13 = theta_13.cos();
        let theta_12 = if cos_13 > 1e-15 {
            (v_us / cos_13).min(1.0).asin()
        } else {
            0.0
        };
        let theta_23 = if cos_13 > 1e-15 {
            (v_cb / cos_13).min(1.0).asin()
        } else {
            0.0
        };

        // Jarlskog: J = Im(V_us * V_cb * V_ub* * V_cs*)
        // For real CKM: J = V[0][0]*V[1][1]*V[0][1]*V[1][0] (approximation)
        let j =
            (v_ckm[(0, 1)] * v_ckm[(1, 2)] * v_ckm[(0, 0)] * v_ckm[(1, 1)]).abs() * theta_13.sin(); // include the sin(theta_13) factor

        println!("  CKM angles (permutation-aligned):");
        println!(
            "    theta_12 = {:.4} deg (PDG: {PDG_CKM_THETA12_DEG})",
            theta_12.to_degrees()
        );
        println!(
            "    theta_13 = {:.4} deg (PDG: {PDG_CKM_THETA13_DEG})",
            theta_13.to_degrees()
        );
        println!(
            "    theta_23 = {:.4} deg (PDG: {PDG_CKM_THETA23_DEG})",
            theta_23.to_degrees()
        );
        println!("    |J| ~ {:.6e} (PDG: {PDG_CKM_JARLSKOG})", j);

        // Mass ratios
        up_evals.sort_by(|a, b| a.abs().partial_cmp(&b.abs()).unwrap());
        down_evals.sort_by(|a, b| a.abs().partial_cmp(&b.abs()).unwrap());
        if down_evals[0].abs() > 1e-10 {
            println!(
                "  m_u/m_d = {:.4} (PDG: {PDG_MU_MD})",
                up_evals[0].abs() / down_evals[0].abs()
            );
        }
        if down_evals[1].abs() > 1e-10 {
            println!(
                "  m_c/m_s = {:.4} (PDG: {PDG_MC_MS})",
                up_evals[1].abs() / down_evals[1].abs()
            );
        }
        if down_evals[2].abs() > 1e-10 {
            println!(
                "  m_t/m_b = {:.4} (PDG: {PDG_MT_MB})",
                up_evals[2].abs() / down_evals[2].abs()
            );
        }
    }

    /// Exhaustive CKM selector pair scan (Rayon-parallelized).
    ///
    /// Scans all splitting-pair combinations for up/down sector assignment.
    /// For each (up_pair, down_pair) combo where up != down, computes
    /// permutation-aligned CKM and measures log-distance to PDG.
    ///
    /// PDG targets: |V_us|=0.2250, |V_ub|=0.00373, |V_cb|=0.0418
    #[test]
    fn test_ckm_selector_pair_scan() {
        use rayon::prelude::*;

        let (basis, cs) = standard_basis_and_cs();

        // Enumerate all 1+1+1 splitting pairs
        let mut splitting_pairs: Vec<(usize, usize)> = Vec::new();
        {
            use crate::{
                bell_inequality::SignTableCache, lepton_mass_hierarchy::cd_braid_signed_friction,
                majorana_braiding::MajoranaMode,
                three_fermion_generations::get_sedenion_subalgebras,
            };

            let (o1, o2, o3) = get_sedenion_subalgebras();
            let sign_table = SignTableCache::new(16);

            for i in 1..16_usize {
                for j in (i + 1)..16 {
                    let mi = MajoranaMode {
                        gamma_index: i - 1,
                        cd_basis_index: i,
                        cd_dim: 16,
                    };
                    let mj = MajoranaMode {
                        gamma_index: j - 1,
                        cd_basis_index: j,
                        cd_dim: 16,
                    };
                    let s1 = cd_braid_signed_friction(&mi, &mj, &o1, &sign_table);
                    let s2 = cd_braid_signed_friction(&mi, &mj, &o2, &sign_table);
                    let s3 = cd_braid_signed_friction(&mi, &mj, &o3, &sign_table);
                    if (s1 - s2).abs() > 1e-9 && (s2 - s3).abs() > 1e-9 && (s1 - s3).abs() > 1e-9 {
                        splitting_pairs.push((i, j));
                    }
                }
            }
        }
        println!("--- CKM SELECTOR PAIR SCAN ---");
        println!("Splitting pairs found: {}", splitting_pairs.len());

        // PDG CKM targets
        let pdg_v_us: f64 = 0.2250;
        let pdg_v_ub: f64 = 0.00373;
        let pdg_v_cb: f64 = 0.0418;
        let ln_v_us = pdg_v_us.ln();
        let ln_v_ub = pdg_v_ub.ln();
        let ln_v_cb = pdg_v_cb.ln();

        // Build all (up, down) combos for Rayon parallel iteration
        let combos: Vec<((usize, usize), (usize, usize))> = splitting_pairs
            .iter()
            .flat_map(|&up| {
                splitting_pairs
                    .iter()
                    .filter(move |&&down| down != up)
                    .map(move |&down| (up, down))
            })
            .collect();

        println!("Total combos to evaluate: {}", combos.len());

        // Parallel scan: each combo independently evaluates CKM
        let mut all_results: Vec<(f64, (usize, usize), (usize, usize), (f64, f64, f64))> = combos
            .par_iter()
            .map(|&(up_pair, down_pair)| {
                let (m_up, m_down) = construct_quark_mass_matrices_scan(
                    &basis,
                    &cs,
                    SubalgebraScheme::InterleavedStride,
                    up_pair,
                    down_pair,
                );

                let m_up_sym = (&m_up + m_up.transpose()) * faer::Scale(0.5);
                let m_down_sym = (&m_down + m_down.transpose()) * faer::Scale(0.5);

                let eig_up = m_up_sym.self_adjoint_eigen(Side::Lower).unwrap();
                let eig_down = m_down_sym.self_adjoint_eigen(Side::Lower).unwrap();

                let v_raw = eig_up.U().transpose() * eig_down.U();
                let (v_ckm, _pu, _pd) = extract_ckm_permutation_aware(&v_raw);

                let v_us = v_ckm[(0, 1)].abs();
                let v_ub = v_ckm[(0, 2)].abs();
                let v_cb = v_ckm[(1, 2)].abs();

                let score = if v_us > 1e-15 && v_ub > 1e-15 && v_cb > 1e-15 {
                    (v_us.ln() - ln_v_us).powi(2)
                        + (v_ub.ln() - ln_v_ub).powi(2)
                        + (v_cb.ln() - ln_v_cb).powi(2)
                } else {
                    f64::INFINITY
                };

                (score, up_pair, down_pair, (v_us, v_ub, v_cb))
            })
            .collect();

        all_results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Report best result
        if let Some(&(score, best_up, best_down, (v_us, v_ub, v_cb))) = all_results.first() {
            let theta_13 = v_ub.asin();
            let cos_13 = theta_13.cos();
            let theta_12 = if cos_13 > 1e-15 {
                (v_us / cos_13).min(1.0).asin()
            } else {
                0.0
            };
            let theta_23 = if cos_13 > 1e-15 {
                (v_cb / cos_13).min(1.0).asin()
            } else {
                0.0
            };

            println!("\nBest selector pair combination:");
            println!("  Up-type:   (e_{}, e_{})", best_up.0, best_up.1);
            println!("  Down-type: (e_{}, e_{})", best_down.0, best_down.1);
            println!("  |V_us| = {:.4} (PDG: {:.4})", v_us, pdg_v_us);
            println!("  |V_ub| = {:.6} (PDG: {:.6})", v_ub, pdg_v_ub);
            println!("  |V_cb| = {:.4} (PDG: {:.4})", v_cb, pdg_v_cb);
            println!(
                "  theta_12 = {:.4} deg (PDG: {PDG_CKM_THETA12_DEG})",
                theta_12.to_degrees()
            );
            println!(
                "  theta_13 = {:.4} deg (PDG: {PDG_CKM_THETA13_DEG})",
                theta_13.to_degrees()
            );
            println!(
                "  theta_23 = {:.4} deg (PDG: {PDG_CKM_THETA23_DEG})",
                theta_23.to_degrees()
            );
            println!("  Log-distance score: {:.4}", score);

            // Print full aligned CKM for the best pair
            let (m_up, m_down) = construct_quark_mass_matrices_scan(
                &basis,
                &cs,
                SubalgebraScheme::InterleavedStride,
                best_up,
                best_down,
            );
            let m_up_sym = (&m_up + m_up.transpose()) * faer::Scale(0.5);
            let m_down_sym = (&m_down + m_down.transpose()) * faer::Scale(0.5);
            let eig_up = m_up_sym.self_adjoint_eigen(Side::Lower).unwrap();
            let eig_down = m_down_sym.self_adjoint_eigen(Side::Lower).unwrap();
            let v_raw = eig_up.U().transpose() * eig_down.U();
            let (v_best, pu, pd) = extract_ckm_permutation_aware(&v_raw);

            println!("\n  Best V_CKM (perm up={:?}, down={:?}):", pu, pd);
            for r in 0..3 {
                println!(
                    "    [{:.6}, {:.6}, {:.6}]",
                    v_best[(r, 0)],
                    v_best[(r, 1)],
                    v_best[(r, 2)]
                );
            }
            println!("  PDG |V_CKM|:");
            println!("    [0.9744, 0.2250, 0.0037]");
            println!("    [0.2249, 0.9735, 0.0418]");
            println!("    [0.0086, 0.0411, 0.9991]");
        }

        // Print top-5
        println!("\n--- TOP-5 SELECTOR PAIRS ---");
        for (rank, (score, up, down, (v_us, v_ub, v_cb))) in all_results.iter().take(5).enumerate()
        {
            println!(
                "  #{}: up=(e_{},e_{}), down=(e_{},e_{}) | V_us={:.4}, V_ub={:.6}, V_cb={:.4} | score={:.4}",
                rank + 1,
                up.0,
                up.1,
                down.0,
                down.1,
                v_us,
                v_ub,
                v_cb,
                score
            );
        }
    }

    /// CKM selector pair scan using TANG CONTIGUOUS-BLOCK subalgebras.
    ///
    /// Tang's scheme: O1={0..7}, O2={0..3,8..11}, O3={0..3,12..15}.
    /// Shared quaternion = {e_1,e_2,e_3}, so pairs within {1,2,3} are
    /// S3-degenerate and cannot contribute to flavor breaking.
    #[test]
    fn test_ckm_selector_pair_scan_tang_contiguous() {
        use crate::{
            bell_inequality::SignTableCache, lepton_mass_hierarchy::cd_braid_signed_friction,
            majorana_braiding::MajoranaMode, sedenion_subalgebras::get_octonion_subalgebras,
        };
        use rayon::prelude::*;

        let (basis, cs) = standard_basis_and_cs();

        // Enumerate splitting pairs under contiguous-block scheme
        let (o1, o2, o3) = get_octonion_subalgebras();
        let sign_table = SignTableCache::new(16);
        let mut splitting_pairs: Vec<(usize, usize)> = Vec::new();

        for i in 1..16_usize {
            for j in (i + 1)..16 {
                let mi = MajoranaMode {
                    gamma_index: i - 1,
                    cd_basis_index: i,
                    cd_dim: 16,
                };
                let mj = MajoranaMode {
                    gamma_index: j - 1,
                    cd_basis_index: j,
                    cd_dim: 16,
                };
                let s1 = cd_braid_signed_friction(&mi, &mj, &o1, &sign_table);
                let s2 = cd_braid_signed_friction(&mi, &mj, &o2, &sign_table);
                let s3 = cd_braid_signed_friction(&mi, &mj, &o3, &sign_table);
                if (s1 - s2).abs() > 1e-9 && (s2 - s3).abs() > 1e-9 && (s1 - s3).abs() > 1e-9 {
                    splitting_pairs.push((i, j));
                }
            }
        }

        println!("--- CKM SELECTOR PAIR SCAN (TANG CONTIGUOUS-BLOCK) ---");
        println!(
            "Splitting pairs found: {} (vs 21 interleaved)",
            splitting_pairs.len()
        );

        if splitting_pairs.is_empty() {
            println!("  NO splitting pairs under contiguous-block scheme!");
            println!("  All friction triples are S3-degenerate.");
            println!("  This means the contiguous-block scheme does NOT break S3.");
            return;
        }

        let pdg_v_us: f64 = 0.2250;
        let pdg_v_ub: f64 = 0.00373;
        let pdg_v_cb: f64 = 0.0418;
        let ln_v_us = pdg_v_us.ln();
        let ln_v_ub = pdg_v_ub.ln();
        let ln_v_cb = pdg_v_cb.ln();

        let combos: Vec<((usize, usize), (usize, usize))> = splitting_pairs
            .iter()
            .flat_map(|&up| {
                splitting_pairs
                    .iter()
                    .filter(move |&&down| down != up)
                    .map(move |&down| (up, down))
            })
            .collect();

        println!("Total combos: {}", combos.len());

        let mut all_results: Vec<(f64, (usize, usize), (usize, usize), (f64, f64, f64))> = combos
            .par_iter()
            .map(|&(up_pair, down_pair)| {
                let (m_up, m_down) = construct_quark_mass_matrices_scan(
                    &basis,
                    &cs,
                    SubalgebraScheme::ContiguousBlock,
                    up_pair,
                    down_pair,
                );
                let m_up_sym = (&m_up + m_up.transpose()) * faer::Scale(0.5);
                let m_down_sym = (&m_down + m_down.transpose()) * faer::Scale(0.5);
                let eig_up = m_up_sym.self_adjoint_eigen(Side::Lower).unwrap();
                let eig_down = m_down_sym.self_adjoint_eigen(Side::Lower).unwrap();
                let v_raw = eig_up.U().transpose() * eig_down.U();
                let (v_ckm, _, _) = extract_ckm_permutation_aware(&v_raw);
                let v_us = v_ckm[(0, 1)].abs();
                let v_ub = v_ckm[(0, 2)].abs();
                let v_cb = v_ckm[(1, 2)].abs();
                let score = if v_us > 1e-15 && v_ub > 1e-15 && v_cb > 1e-15 {
                    (v_us.ln() - ln_v_us).powi(2)
                        + (v_ub.ln() - ln_v_ub).powi(2)
                        + (v_cb.ln() - ln_v_cb).powi(2)
                } else {
                    f64::INFINITY
                };
                (score, up_pair, down_pair, (v_us, v_ub, v_cb))
            })
            .collect();

        all_results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        if let Some(&(score, best_up, best_down, (v_us, v_ub, v_cb))) = all_results.first() {
            let theta_13 = v_ub.asin();
            let cos_13 = theta_13.cos();
            let theta_12 = if cos_13 > 1e-15 {
                (v_us / cos_13).min(1.0).asin()
            } else {
                0.0
            };
            let theta_23 = if cos_13 > 1e-15 {
                (v_cb / cos_13).min(1.0).asin()
            } else {
                0.0
            };

            println!("\nBest Tang contiguous-block CKM:");
            println!("  Up-type:   (e_{}, e_{})", best_up.0, best_up.1);
            println!("  Down-type: (e_{}, e_{})", best_down.0, best_down.1);
            println!("  |V_us| = {:.4} (PDG: {:.4})", v_us, pdg_v_us);
            println!("  |V_ub| = {:.6} (PDG: {:.6})", v_ub, pdg_v_ub);
            println!("  |V_cb| = {:.4} (PDG: {:.4})", v_cb, pdg_v_cb);
            println!("  theta_12 = {:.4} deg (PDG: 12.99)", theta_12.to_degrees());
            println!("  theta_13 = {:.4} deg (PDG: 0.214)", theta_13.to_degrees());
            println!("  theta_23 = {:.4} deg (PDG: 2.40)", theta_23.to_degrees());
            println!("  Score: {:.4} (interleaved best: 0.0102)", score);
        }

        println!("\n--- TOP-5 TANG CONTIGUOUS PAIRS ---");
        for (rank, (score, up, down, (v_us, v_ub, v_cb))) in all_results.iter().take(5).enumerate()
        {
            println!(
                "  #{}: up=(e_{},e_{}), down=(e_{},e_{}) | V_us={:.4}, V_ub={:.6}, V_cb={:.4} | score={:.4}",
                rank + 1,
                up.0,
                up.1,
                down.0,
                down.1,
                v_us,
                v_ub,
                v_cb,
                score
            );
        }
    }

    /// Regression test: verify CasimirBaseline refactor preserves quark results.
    ///
    /// Pins CKM angles and mass eigenvalues to pre-refactor values. If this test
    /// fails after a Casimir refactor, the internal restructuring changed behavior.
    #[test]
    fn test_ckm_casimir_refactor_regression() {
        let (basis, cs) = standard_basis_and_cs();
        let scheme = SubalgebraScheme::InterleavedStride;

        // Verify CasimirBaseline decomposition: c_su3 + c_su2 = M_up, c_su3 - c_su2 = M_down
        let cb = construct_casimir_projections(&basis, &cs, scheme);
        let (m_up_via_wrapper, m_down_via_wrapper) =
            construct_quark_mass_matrices(&basis, &cs, scheme);
        let (m_up_via_assemble, m_down_via_assemble) = assemble_quark_matrices(&cb);

        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (m_up_via_wrapper[(i, j)] - m_up_via_assemble[(i, j)]).abs() < 1e-14,
                    "M_up mismatch at ({},{})",
                    i,
                    j
                );
                assert!(
                    (m_down_via_wrapper[(i, j)] - m_down_via_assemble[(i, j)]).abs() < 1e-14,
                    "M_down mismatch at ({},{})",
                    i,
                    j
                );
            }
        }

        // Verify the raw components reconstruct correctly
        for i in 0..3 {
            for j in 0..3 {
                let reconstructed_up = cb.c_su3[(i, j)] + cb.c_su2[(i, j)];
                let reconstructed_down = cb.c_su3[(i, j)] - cb.c_su2[(i, j)];
                assert!(
                    (reconstructed_up - m_up_via_wrapper[(i, j)]).abs() < 1e-14,
                    "c_su3 + c_su2 != M_up at ({},{})",
                    i,
                    j
                );
                assert!(
                    (reconstructed_down - m_down_via_wrapper[(i, j)]).abs() < 1e-14,
                    "c_su3 - c_su2 != M_down at ({},{})",
                    i,
                    j
                );
            }
        }

        // Pin CKM angles from current sort_mass_eigenstates ordering.
        // Note: derive_ckm_matrix does not apply extract_ckm_permutation_aware,
        // so angles are read directly from the raw V = U_up^T U_down matrix.
        // After sort_mass_eigenstates refactor (ascending |mass|), the (0,1)
        // element is large, giving theta_12 = 56.39 deg.
        let ckm = derive_ckm_matrix(&basis, &cs, scheme);
        assert!(
            (ckm.angles_deg.0 - 56.39).abs() < 0.01,
            "CKM theta_12 regression: got {:.4}",
            ckm.angles_deg.0
        );
        assert!(
            ckm.angles_deg.1.abs() < 0.01,
            "CKM theta_13 regression: got {:.4}",
            ckm.angles_deg.1
        );
        assert!(
            ckm.angles_deg.2.abs() < 0.01,
            "CKM theta_23 regression: got {:.4}",
            ckm.angles_deg.2
        );

        // Pin mass eigenvalues (up sector dominant: 13.6875)
        assert!(
            (ckm.up_masses[2] - 13.6875).abs() < 0.001,
            "up mass[2] regression: got {:.6}",
            ckm.up_masses[2]
        );
        assert!(
            (ckm.down_masses[2] - 10.3125).abs() < 0.001,
            "down mass[2] regression: got {:.6}",
            ckm.down_masses[2]
        );

        println!("PASS: CKM Casimir refactor regression");
    }

    /// 3-blade quark mass ratios: scan triples for m_u/m_d, m_c/m_s, m_t/m_b.
    ///
    /// PDG 2024 mass ratios (at 2 GeV):
    ///   m_u/m_d ~ 0.47, m_c/m_s ~ 11.8, m_t/m_b ~ 41.3
    ///   m_c/m_u ~ 550, m_t/m_c ~ 130, m_b/m_s ~ 51.5
    #[test]
    fn test_3blade_quark_mass_ratios() {
        use crate::{
            bell_inequality::SignTableCache, lepton_mass_hierarchy::cd_braid_signed_friction,
            majorana_braiding::MajoranaMode, three_fermion_generations::get_sedenion_subalgebras,
        };
        use rayon::prelude::*;

        let (o1, o2, o3) = get_sedenion_subalgebras();
        let subs = [o1.clone(), o2.clone(), o3.clone()];

        let w1 = -0.656850_f64;
        let w2 = -0.741999_f64;

        // PDG quark mass ratios (target)
        let pdg_mc_mu = 550.0_f64;
        let pdg_mt_mc = 130.0_f64;

        let mut triples: Vec<(usize, usize, usize)> = Vec::new();
        for i in 1..16_usize {
            for j in (i + 1)..16 {
                for k in (j + 1)..16 {
                    triples.push((i, j, k));
                }
            }
        }

        let compute_3blade = |triple: (usize, usize, usize)| -> [f64; 3] {
            let sign_table = SignTableCache::new(16);
            let (a, b, c) = triple;
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
            let mut f = [0.0_f64; 3];
            for (g, sub) in subs.iter().enumerate() {
                f[g] = cd_braid_signed_friction(&ma, &mb, sub, &sign_table)
                    + cd_braid_signed_friction(&ma, &mc, sub, &sign_table)
                    + cd_braid_signed_friction(&mb, &mc, sub, &sign_table);
            }
            f
        };

        let all_frictions: Vec<[f64; 3]> = triples.par_iter().map(|&t| compute_3blade(t)).collect();

        // Scan up-type triple x down-type triple
        let af = &all_frictions;
        let results: Vec<_> = (0..triples.len())
            .into_par_iter()
            .flat_map_iter(|ui| {
                (0..triples.len()).map(move |di| {
                    let sel_up = &af[ui];
                    let sel_dn = &af[di];
                    let mut masses_up = [0.0_f64; 3];
                    let mut masses_dn = [0.0_f64; 3];
                    for g in 0..3 {
                        masses_up[g] = (w1 * sel_up[g] + w2 * sel_dn[g]).exp();
                        masses_dn[g] = (w1 * sel_dn[g] + w2 * sel_up[g]).exp();
                    }
                    masses_up.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    masses_dn.sort_by(|a, b| a.partial_cmp(b).unwrap());

                    // Ratios: m_c/m_u = m2/m1 (up), m_t/m_c = m3/m2 (up)
                    let mc_mu = if masses_up[0] > 1e-30 {
                        masses_up[1] / masses_up[0]
                    } else {
                        f64::MAX
                    };
                    let mt_mc = if masses_up[1] > 1e-30 {
                        masses_up[2] / masses_up[1]
                    } else {
                        f64::MAX
                    };

                    let err = ((mc_mu - pdg_mc_mu) / pdg_mc_mu).powi(2)
                        + ((mt_mc - pdg_mt_mc) / pdg_mt_mc).powi(2);

                    (err, mc_mu, mt_mc, ui, di, masses_up[2] / masses_up[0])
                })
            })
            .collect();

        let mut sorted = results;
        sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        println!("  === 3-Blade Quark Mass Ratios ===\n");
        println!("  PDG: m_c/m_u ~ 550, m_t/m_c ~ 130\n");
        println!(
            "  {:>5} | {:>12} {:>12} | {:>8} {:>8} | {:>8}",
            "rank", "up_triple", "dn_triple", "mc/mu", "mt/mc", "mt/mu"
        );

        for (rank, e) in sorted.iter().take(10).enumerate() {
            println!(
                "  {:>5} | {:>12?} {:>12?} | {:>8.1} {:>8.1} | {:>8.0}",
                rank + 1,
                triples[e.3],
                triples[e.4],
                e.1,
                e.2,
                e.5
            );
        }

        let best = &sorted[0];
        println!(
            "\n  BEST: m_c/m_u = {:.1} (PDG: 550), m_t/m_c = {:.1} (PDG: 130)",
            best.1, best.2
        );
        println!("  m_t/m_u = {:.0} (PDG: ~71500)", best.5);
        println!(
            "  up_triple = {:?}, dn_triple = {:?}",
            triples[best.3], triples[best.4]
        );

        // Also compute down-type ratios for the best pair
        let sel_up = &af[best.3];
        let sel_dn = &af[best.4];
        let mut masses_dn = [0.0_f64; 3];
        for g in 0..3 {
            masses_dn[g] = (w1 * sel_dn[g] + w2 * sel_up[g]).exp();
        }
        masses_dn.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let ms_md = if masses_dn[0] > 1e-30 {
            masses_dn[1] / masses_dn[0]
        } else {
            f64::MAX
        };
        let mb_ms = if masses_dn[1] > 1e-30 {
            masses_dn[2] / masses_dn[1]
        } else {
            f64::MAX
        };
        let mb_md = if masses_dn[0] > 1e-30 {
            masses_dn[2] / masses_dn[0]
        } else {
            f64::MAX
        };

        println!("\n  Down-type ratios (same pair):");
        println!("    m_s/m_d = {:.1} (PDG: ~20)", ms_md);
        println!("    m_b/m_s = {:.1} (PDG: ~51.5)", mb_ms);
        println!("    m_b/m_d = {:.0} (PDG: ~1030)", mb_md);
    }

    /// 3-blade down-type quark mass ratios (separate optimization).
    ///
    /// PDG: m_s/m_d ~ 20, m_b/m_s ~ 51.5.
    #[test]
    fn test_3blade_down_quark_mass_ratios() {
        use crate::{
            bell_inequality::SignTableCache, lepton_mass_hierarchy::cd_braid_signed_friction,
            majorana_braiding::MajoranaMode, three_fermion_generations::get_sedenion_subalgebras,
        };
        use rayon::prelude::*;

        let (o1, o2, o3) = get_sedenion_subalgebras();
        let subs = [o1.clone(), o2.clone(), o3.clone()];
        let w1 = -0.656850_f64;
        let w2 = -0.741999_f64;
        let pdg_ms_md = 20.0_f64;
        let pdg_mb_ms = 51.5_f64;

        let mut triples: Vec<(usize, usize, usize)> = Vec::new();
        for i in 1..16_usize {
            for j in (i + 1)..16 {
                for k in (j + 1)..16 {
                    triples.push((i, j, k));
                }
            }
        }

        let compute_3blade = |triple: (usize, usize, usize)| -> [f64; 3] {
            let sign_table = SignTableCache::new(16);
            let (a, b, c) = triple;
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
            let mut f = [0.0_f64; 3];
            for (g, sub) in subs.iter().enumerate() {
                f[g] = cd_braid_signed_friction(&ma, &mb, sub, &sign_table)
                    + cd_braid_signed_friction(&ma, &mc, sub, &sign_table)
                    + cd_braid_signed_friction(&mb, &mc, sub, &sign_table);
            }
            f
        };

        let all_frictions: Vec<[f64; 3]> = triples.par_iter().map(|&t| compute_3blade(t)).collect();

        // Optimize for DOWN-type ratios: F_dn = w1*sel_dn + w2*sel_up
        let af = &all_frictions;
        let results: Vec<_> = (0..triples.len())
            .into_par_iter()
            .flat_map_iter(|di| {
                (0..triples.len()).map(move |ui| {
                    let sel_dn = &af[di];
                    let sel_up = &af[ui];
                    let mut masses = [0.0_f64; 3];
                    for g in 0..3 {
                        masses[g] = (w1 * sel_dn[g] + w2 * sel_up[g]).exp();
                    }
                    masses.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let ms_md = if masses[0] > 1e-30 {
                        masses[1] / masses[0]
                    } else {
                        f64::MAX
                    };
                    let mb_ms = if masses[1] > 1e-30 {
                        masses[2] / masses[1]
                    } else {
                        f64::MAX
                    };
                    let err = ((ms_md - pdg_ms_md) / pdg_ms_md).powi(2)
                        + ((mb_ms - pdg_mb_ms) / pdg_mb_ms).powi(2);
                    (err, ms_md, mb_ms, di, ui)
                })
            })
            .collect();

        let mut sorted = results;
        sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        println!("  === 3-Blade Down-Type Quark Mass Ratios ===\n");
        println!("  PDG: m_s/m_d ~ 20, m_b/m_s ~ 51.5\n");
        for (rank, e) in sorted.iter().take(5).enumerate() {
            println!(
                "  {:>3}. dn={:?} up={:?}: m_s/m_d={:.1}, m_b/m_s={:.1}",
                rank + 1,
                triples[e.3],
                triples[e.4],
                e.1,
                e.2
            );
        }
        let best = &sorted[0];
        println!(
            "\n  BEST: m_s/m_d = {:.1} (PDG: 20, err: {:.1}%)",
            best.1,
            ((best.1 - pdg_ms_md) / pdg_ms_md * 100.0).abs()
        );
        println!(
            "  m_b/m_s = {:.1} (PDG: 51.5, err: {:.1}%)",
            best.2,
            ((best.2 - pdg_mb_ms) / pdg_mb_ms * 100.0).abs()
        );
    }

    /// 3-blade charged lepton mass ratios with universal (w1, w2).
    ///
    /// PDG: m_mu/m_e ~ 207, m_tau/m_e ~ 3477.
    #[test]
    fn test_3blade_lepton_mass_ratios() {
        use crate::{
            bell_inequality::SignTableCache, lepton_mass_hierarchy::cd_braid_signed_friction,
            majorana_braiding::MajoranaMode, three_fermion_generations::get_sedenion_subalgebras,
        };
        use rayon::prelude::*;

        let (o1, o2, o3) = get_sedenion_subalgebras();
        let subs = [o1.clone(), o2.clone(), o3.clone()];
        let w1 = -0.656850_f64;
        let w2 = -0.741999_f64;

        let pdg_mmu_me = 206.768_f64;
        let pdg_mtau_me = 3477.2_f64;

        let mut triples: Vec<(usize, usize, usize)> = Vec::new();
        for i in 1..16_usize {
            for j in (i + 1)..16 {
                for k in (j + 1)..16 {
                    triples.push((i, j, k));
                }
            }
        }

        let compute_3blade = |triple: (usize, usize, usize)| -> [f64; 3] {
            let sign_table = SignTableCache::new(16);
            let (a, b, c) = triple;
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
            let mut f = [0.0_f64; 3];
            for (g, sub) in subs.iter().enumerate() {
                f[g] = cd_braid_signed_friction(&ma, &mb, sub, &sign_table)
                    + cd_braid_signed_friction(&ma, &mc, sub, &sign_table)
                    + cd_braid_signed_friction(&mb, &mc, sub, &sign_table);
            }
            f
        };

        let all_frictions: Vec<[f64; 3]> = triples.par_iter().map(|&t| compute_3blade(t)).collect();

        // For leptons: single selector (no cross-coupling), so m_g ~ exp(w1 * f_g)
        // Actually the lepton model uses TWO selectors with cross-coupling,
        // same as quarks. Scan pairs of triples.
        let af = &all_frictions;
        let results: Vec<_> = (0..triples.len())
            .into_par_iter()
            .flat_map_iter(|ci| {
                (0..triples.len()).map(move |ni| {
                    let sel_ch = &af[ci];
                    let sel_nu = &af[ni];
                    let mut masses = [0.0_f64; 3];
                    for g in 0..3 {
                        masses[g] = (w1 * sel_ch[g] + w2 * sel_nu[g]).exp();
                    }
                    masses.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let mmu_me = if masses[0] > 1e-30 {
                        masses[1] / masses[0]
                    } else {
                        f64::MAX
                    };
                    let mtau_me = if masses[0] > 1e-30 {
                        masses[2] / masses[0]
                    } else {
                        f64::MAX
                    };
                    let err = ((mmu_me - pdg_mmu_me) / pdg_mmu_me).powi(2)
                        + ((mtau_me - pdg_mtau_me) / pdg_mtau_me).powi(2);
                    (err, mmu_me, mtau_me, ci, ni)
                })
            })
            .collect();

        let mut sorted = results;
        sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        println!("  === 3-Blade Lepton Mass Ratios ===\n");
        println!("  PDG: m_mu/m_e ~ 207, m_tau/m_e ~ 3477\n");
        for (rank, e) in sorted.iter().take(5).enumerate() {
            println!(
                "  {:>3}. ch={:?} nu={:?}: m_mu/m_e={:.1}, m_tau/m_e={:.0}",
                rank + 1,
                triples[e.3],
                triples[e.4],
                e.1,
                e.2
            );
        }

        let best = &sorted[0];
        println!(
            "\n  BEST: m_mu/m_e = {:.1} (PDG: 207, err: {:.1}%)",
            best.1,
            ((best.1 - pdg_mmu_me) / pdg_mmu_me * 100.0).abs()
        );
        println!(
            "  m_tau/m_e = {:.0} (PDG: 3477, err: {:.1}%)",
            best.2,
            ((best.2 - pdg_mtau_me) / pdg_mtau_me * 100.0).abs()
        );
    }
}
