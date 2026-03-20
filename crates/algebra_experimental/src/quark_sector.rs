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

use crate::cayley_dickson_structs::Sedenion;
use crate::quantum_state::QuantumState;
use crate::su_n_generators::construct_su5_generators_algebraic;
use crate::neutrino_sector::{classify_generator, GeneratorType};
use crate::sedenion_subalgebras::get_octonion_subalgebras;
use crate::three_fermion_generations::get_sedenion_subalgebras;
use faer::{Mat, Side};

/// Configuration for which subalgebra definition to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubalgebraScheme {
    /// Contiguous CD-doubling blocks: O1=[0..7], O2=[0,1,2,3,8..11], O3=[0,1,2,3,12..15]
    ContiguousBlock,
    /// Interleaved stride: O1=[0,1,4,5,8,9,12,13], etc.
    InterleavedStride,
}

/// Get the three octonionic subalgebra index sets for a given scheme.
pub fn get_subalgebras(scheme: SubalgebraScheme) -> [Vec<usize>; 3] {
    match scheme {
        SubalgebraScheme::ContiguousBlock => {
            let (o1, o2, o3) = get_octonion_subalgebras();
            [o1, o2, o3]
        }
        SubalgebraScheme::InterleavedStride => {
            let (o1, o2, o3) = get_sedenion_subalgebras();
            [o1, o2, o3]
        }
    }
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
        .filter(|(i, g)| classify_generator(*i) == GeneratorType::SU3 && **g != QuantumState::TopologicalNull)
        .map(|(_, g)| *g)
        .collect();

    // Extract SU(2) generators (indices 8-10)
    let su2_gens: Vec<QuantumState> = su5_gens
        .iter()
        .enumerate()
        .filter(|(i, g)| classify_generator(*i) == GeneratorType::SU2 && **g != QuantumState::TopologicalNull)
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
/// Returns (M_up, M_down) as 3x3 faer::Mat<f64>.
pub fn construct_quark_mass_matrices(
    basis: &[Sedenion; 16],
    complex_structure: &Sedenion,
    scheme: SubalgebraScheme,
) -> (Mat<f64>, Mat<f64>) {
    let su5_gens = construct_su5_generators_algebraic(basis, complex_structure);
    let subalgebras = get_subalgebras(scheme);

    // SU(3) Casimir: C_2 = sum_{a=0}^{7} T_a * T_a
    let su3_gens: Vec<QuantumState> = su5_gens
        .iter()
        .enumerate()
        .filter(|(i, g)| classify_generator(*i) == GeneratorType::SU3 && **g != QuantumState::TopologicalNull)
        .map(|(_, g)| *g)
        .collect();

    let casimir_su3 = su3_gens
        .iter()
        .fold(QuantumState::Observable(Sedenion::default()), |acc, g| acc + *g * *g);

    let casimir_s = match casimir_su3 {
        QuantumState::Observable(s) => s,
        QuantumState::TopologicalNull => Sedenion::default(),
    };

    // SU(2) Casimir for up/down separation
    let su2_gens: Vec<QuantumState> = su5_gens
        .iter()
        .enumerate()
        .filter(|(i, g)| classify_generator(*i) == GeneratorType::SU2 && **g != QuantumState::TopologicalNull)
        .map(|(_, g)| *g)
        .collect();

    let casimir_su2 = su2_gens
        .iter()
        .fold(QuantumState::Observable(Sedenion::default()), |acc, g| acc + *g * *g);

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

    // Build 3x3 mass matrices
    // M_up_{ij} = Re(proj_su3[i].conj() * proj_su3[j] + proj_su2[i].conj() * proj_su2[j])
    // M_down_{ij} = Re(proj_su3[i].conj() * proj_su3[j] - proj_su2[i].conj() * proj_su2[j])
    let mut m_up = Mat::<f64>::zeros(3, 3);
    let mut m_down = Mat::<f64>::zeros(3, 3);

    for i in 0..3 {
        for j in 0..3 {
            let su3_term = (proj_su3[i].conj() * proj_su3[j]).to_slice()[0];
            let su2_term = (proj_su2[i].conj() * proj_su2[j]).to_slice()[0];

            m_up.write(i, j, su3_term + su2_term);
            m_down.write(i, j, su3_term - su2_term);
        }
    }

    (m_up, m_down)
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

/// Derive CKM matrix: V_CKM = U_up^T * U_down.
pub fn derive_ckm_matrix(
    basis: &[Sedenion; 16],
    complex_structure: &Sedenion,
    scheme: SubalgebraScheme,
) -> CkmResult {
    let (m_up, m_down) = construct_quark_mass_matrices(basis, complex_structure, scheme);

    // Symmetrize (numerical safety)
    let m_up_sym = (&m_up + m_up.transpose()) * faer::scale(0.5);
    let m_down_sym = (&m_down + m_down.transpose()) * faer::scale(0.5);

    let eig_up = m_up_sym.selfadjoint_eigendecomposition(Side::Lower);
    let eig_down = m_down_sym.selfadjoint_eigendecomposition(Side::Lower);

    let s_up = eig_up.s();
    let u_up = eig_up.u();
    let s_down = eig_down.s();
    let u_down = eig_down.u();

    // Extract eigenvalues
    let mut up_masses = [0.0; 3];
    let mut down_masses = [0.0; 3];
    for i in 0..3 {
        up_masses[i] = s_up.column_vector().read(i).abs();
        down_masses[i] = s_down.column_vector().read(i).abs();
    }

    // V_CKM = U_up^T * U_down
    let v_ckm = u_up.transpose() * u_down;

    // Extract angles from the standard parameterization:
    //   V_us = sin(theta_12)*cos(theta_13)
    //   V_ub = sin(theta_13)
    //   V_cb = sin(theta_23)*cos(theta_13)
    let v_us = v_ckm.read(0, 1).abs();
    let v_ub = v_ckm.read(0, 2).abs();
    let v_cb = v_ckm.read(1, 2).abs();
    let _v_ud = v_ckm.read(0, 0).abs();
    let _v_tb = v_ckm.read(2, 2).abs();

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
    let v_cs = v_ckm.read(1, 1);
    let j_invariant = v_ckm.read(0, 1) * v_ckm.read(1, 2) * v_ckm.read(0, 2) * v_cs;
    let cp_phase_deg = if j_invariant.abs() > 1e-15 {
        j_invariant.asin().to_degrees()
    } else {
        0.0
    };

    // Sort masses ascending
    up_masses.sort_by(|a, b| a.partial_cmp(b).unwrap());
    down_masses.sort_by(|a, b| a.partial_cmp(b).unwrap());

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
    let m_u_over_m_d = if down[0] > 1e-15 { up[0] / down[0] } else { f64::INFINITY };
    let m_c_over_m_s = if down[1] > 1e-15 { up[1] / down[1] } else { f64::INFINITY };
    let m_t_over_m_b = if down[2] > 1e-15 { up[2] / down[2] } else { f64::INFINITY };

    QuarkMassRatios {
        m_u_over_m_d,
        m_c_over_m_s,
        m_t_over_m_b,
        scheme,
    }
}

/// PDG mass ratio targets at M_Z scale.
const PDG_MU_MD: f64 = 0.47;
const PDG_MC_MS: f64 = 11.7;
const PDG_MT_MB: f64 = 40.8;

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
pub fn compare_schemes(
    basis: &[Sedenion; 16],
    complex_structure: &Sedenion,
) -> SchemeComparison {
    let ckm_c = derive_ckm_matrix(basis, complex_structure, SubalgebraScheme::ContiguousBlock);
    let ckm_i = derive_ckm_matrix(basis, complex_structure, SubalgebraScheme::InterleavedStride);

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

        for scheme in [SubalgebraScheme::ContiguousBlock, SubalgebraScheme::InterleavedStride] {
            let (up, down) = construct_quark_ladder_operators(&basis, &cs, scheme);

            println!("--- QUARK LADDER OPERATORS ({scheme:?}) ---");
            println!("  up-type count: {}", up.len());
            println!("  down-type count: {}", down.len());

            let up_non_null = up.iter().filter(|q| **q != QuantumState::TopologicalNull).count();
            let down_non_null = down.iter().filter(|q| **q != QuantumState::TopologicalNull).count();
            println!("  up-type non-null: {up_non_null}");
            println!("  down-type non-null: {down_non_null}");

            assert!(!up.is_empty(), "{scheme:?}: up-type must not be empty");
            assert!(!down.is_empty(), "{scheme:?}: down-type must not be empty");
        }
    }

    #[test]
    fn test_quark_mass_matrices_symmetric() {
        let (basis, cs) = standard_basis_and_cs();

        for scheme in [SubalgebraScheme::ContiguousBlock, SubalgebraScheme::InterleavedStride] {
            let (m_up, m_down) = construct_quark_mass_matrices(&basis, &cs, scheme);

            println!("--- QUARK MASS MATRICES ({scheme:?}) ---");
            println!("  M_up = {:?}", m_up);
            println!("  M_down = {:?}", m_down);

            // Check symmetry
            for i in 0..3 {
                for j in 0..3 {
                    assert!(
                        (m_up.read(i, j) - m_up.read(j, i)).abs() < 1e-12,
                        "{scheme:?}: M_up not symmetric at ({i},{j})"
                    );
                    assert!(
                        (m_down.read(i, j) - m_down.read(j, i)).abs() < 1e-12,
                        "{scheme:?}: M_down not symmetric at ({i},{j})"
                    );
                }
            }
        }
    }

    #[test]
    fn test_ckm_unitarity() {
        let (basis, cs) = standard_basis_and_cs();

        for scheme in [SubalgebraScheme::ContiguousBlock, SubalgebraScheme::InterleavedStride] {
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
                        (vvt.read(i, j) - expected).abs() < 1e-10,
                        "{scheme:?}: V*V^T not identity at ({i},{j}): got {:.6e}",
                        vvt.read(i, j)
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

        // Both schemes must produce finite positive ratios
        for ratios in [&comparison.contiguous, &comparison.interleaved] {
            assert!(
                ratios.m_u_over_m_d.is_finite(),
                "{:?}: m_u/m_d not finite", ratios.scheme
            );
            assert!(
                ratios.m_c_over_m_s.is_finite(),
                "{:?}: m_c/m_s not finite", ratios.scheme
            );
            assert!(
                ratios.m_t_over_m_b.is_finite(),
                "{:?}: m_t/m_b not finite", ratios.scheme
            );
        }
    }
}
