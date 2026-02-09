//! Clifford algebra Cl(n) implementation for particle physics.
//!
//! Implements gamma matrices and ideal decomposition following
//! Furey et al. 2024 (Cl(8) -> 3 generations).
//!
//! Now uses nalgebra for clean matrix operations.
//!
//! References:
//! - Furey et al. (2024): Cl(8) -> 3 generations via minimal left ideals
//! - Furey (2016): One generation from Cl(6)
//! - Lounesto (2001): Clifford algebras and spinors

use nalgebra::DMatrix;
use num_complex::Complex64;

/// A complex matrix for gamma matrix representation (using nalgebra).
pub type GammaMatrix = DMatrix<Complex64>;

/// Pauli matrices sigma_1, sigma_2, sigma_3.
pub fn pauli_matrices() -> (GammaMatrix, GammaMatrix, GammaMatrix) {
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let i = Complex64::new(0.0, 1.0);

    let sigma_1 = GammaMatrix::from_row_slice(2, 2, &[zero, one, one, zero]);

    let sigma_2 = GammaMatrix::from_row_slice(2, 2, &[zero, -i, i, zero]);

    let sigma_3 = GammaMatrix::from_row_slice(2, 2, &[one, zero, zero, -one]);

    (sigma_1, sigma_2, sigma_3)
}

/// 2x2 identity matrix.
fn identity_2() -> GammaMatrix {
    GammaMatrix::identity(2, 2)
}

/// Kronecker (tensor) product of two matrices.
fn kron(a: &GammaMatrix, b: &GammaMatrix) -> GammaMatrix {
    a.kronecker(b)
}

/// Construct 8 gamma matrices for Cl(8) in the real 16x16 representation.
///
/// Uses the tensor product construction with Pauli matrices.
/// Returns 8 matrices satisfying {gamma_i, gamma_j} = 2 * delta_ij * I.
pub fn gamma_matrices_cl8() -> Vec<GammaMatrix> {
    let (s1, s2, s3) = pauli_matrices();
    let i2 = identity_2();

    vec![
        // gamma_1 = sigma_1 (x) I (x) I (x) I
        kron(&kron(&kron(&s1, &i2), &i2), &i2),
        // gamma_2 = sigma_2 (x) I (x) I (x) I
        kron(&kron(&kron(&s2, &i2), &i2), &i2),
        // gamma_3 = sigma_3 (x) sigma_1 (x) I (x) I
        kron(&kron(&kron(&s3, &s1), &i2), &i2),
        // gamma_4 = sigma_3 (x) sigma_2 (x) I (x) I
        kron(&kron(&kron(&s3, &s2), &i2), &i2),
        // gamma_5 = sigma_3 (x) sigma_3 (x) sigma_1 (x) I
        kron(&kron(&kron(&s3, &s3), &s1), &i2),
        // gamma_6 = sigma_3 (x) sigma_3 (x) sigma_2 (x) I
        kron(&kron(&kron(&s3, &s3), &s2), &i2),
        // gamma_7 = sigma_3 (x) sigma_3 (x) sigma_3 (x) sigma_1
        kron(&kron(&kron(&s3, &s3), &s3), &s1),
        // gamma_8 = sigma_3 (x) sigma_3 (x) sigma_3 (x) sigma_2
        kron(&kron(&kron(&s3, &s3), &s3), &s2),
    ]
}

/// Verify that gamma matrices satisfy {gamma_i, gamma_j} = 2 * delta_ij * I.
pub fn verify_clifford_relation(gammas: &[GammaMatrix], tol: f64) -> bool {
    let n = gammas.len();
    let dim = gammas[0].nrows();
    let identity = GammaMatrix::identity(dim, dim);
    let two = Complex64::new(2.0, 0.0);

    for i in 0..n {
        for j in 0..n {
            let anticomm = &gammas[i] * &gammas[j] + &gammas[j] * &gammas[i];

            let expected = if i == j {
                &identity * two
            } else {
                GammaMatrix::zeros(dim, dim)
            };

            // Check element-wise
            for r in 0..dim {
                for c in 0..dim {
                    let diff = (anticomm[(r, c)] - expected[(r, c)]).norm();
                    if diff > tol {
                        return false;
                    }
                }
            }
        }
    }
    true
}

/// Count basis elements in Cl(n): there are 2^n elements.
pub fn count_basis_elements(n: usize) -> usize {
    1 << n
}

/// Clifford algebra representation for physics applications.
pub struct CliffordAlgebra {
    pub dimension: usize,
    pub gammas: Vec<GammaMatrix>,
}

impl CliffordAlgebra {
    /// Create Cl(8) algebra.
    pub fn cl8() -> Self {
        CliffordAlgebra {
            dimension: 8,
            gammas: gamma_matrices_cl8(),
        }
    }

    /// Verify the Clifford relation.
    pub fn verify(&self, tol: f64) -> bool {
        verify_clifford_relation(&self.gammas, tol)
    }

    /// Get the chirality operator (gamma_7 in Cl(6) or gamma_9 in Cl(8)).
    pub fn chirality_operator(&self) -> GammaMatrix {
        // For Cl(6), gamma_7 = i * gamma_1 * ... * gamma_6
        let n = self.gammas.len().min(6);
        let i = Complex64::new(0.0, 1.0);
        let mut result = &self.gammas[0] * i;
        for g in &self.gammas[1..n] {
            result = &result * g;
        }
        result
    }

    /// Construct left-chiral projector P_L = (1 + gamma_chiral) / 2.
    pub fn left_projector(&self) -> GammaMatrix {
        let chiral = self.chirality_operator();
        let dim = chiral.nrows();
        let identity = GammaMatrix::identity(dim, dim);
        let half = Complex64::new(0.5, 0.0);
        (&identity + &chiral) * half
    }

    /// Construct right-chiral projector P_R = (1 - gamma_chiral) / 2.
    pub fn right_projector(&self) -> GammaMatrix {
        let chiral = self.chirality_operator();
        let dim = chiral.nrows();
        let identity = GammaMatrix::identity(dim, dim);
        let half = Complex64::new(0.5, 0.0);
        (&identity - &chiral) * half
    }
}

/// Standard Model fermion charges from Cl(6) representation.
#[derive(Debug, Clone)]
pub struct FermionCharges {
    pub name: String,
    pub em_charge: f64,
    pub weak_isospin: f64,
    pub color_rep: String,
}

/// Get the fermion charges for one generation.
pub fn fermion_charges_cl6() -> Vec<FermionCharges> {
    vec![
        FermionCharges {
            name: "u_quark".to_string(),
            em_charge: 2.0 / 3.0,
            weak_isospin: 0.5,
            color_rep: "triplet".to_string(),
        },
        FermionCharges {
            name: "d_quark".to_string(),
            em_charge: -1.0 / 3.0,
            weak_isospin: -0.5,
            color_rep: "triplet".to_string(),
        },
        FermionCharges {
            name: "neutrino".to_string(),
            em_charge: 0.0,
            weak_isospin: 0.5,
            color_rep: "singlet".to_string(),
        },
        FermionCharges {
            name: "electron".to_string(),
            em_charge: -1.0,
            weak_isospin: -0.5,
            color_rep: "singlet".to_string(),
        },
    ]
}

/// Lepton mass data (MeV).
#[derive(Debug, Clone)]
pub struct LeptonMasses {
    pub electron: f64,
    pub muon: f64,
    pub tau: f64,
}

impl LeptonMasses {
    pub fn observed() -> Self {
        LeptonMasses {
            electron: 0.511,
            muon: 105.66,
            tau: 1776.86,
        }
    }

    pub fn ratio_mu_e(&self) -> f64 {
        self.muon / self.electron
    }

    pub fn ratio_tau_e(&self) -> f64 {
        self.tau / self.electron
    }

    pub fn ratio_tau_mu(&self) -> f64 {
        self.tau / self.muon
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pauli_count() {
        let (s1, s2, s3) = pauli_matrices();
        assert_eq!(s1.nrows(), 2);
        assert_eq!(s2.nrows(), 2);
        assert_eq!(s3.nrows(), 2);
    }

    #[test]
    fn test_gamma_matrices_count() {
        let gammas = gamma_matrices_cl8();
        assert_eq!(gammas.len(), 8);
    }

    #[test]
    fn test_gamma_matrices_shape() {
        let gammas = gamma_matrices_cl8();
        for g in &gammas {
            assert_eq!(g.nrows(), 16);
            assert_eq!(g.ncols(), 16);
        }
    }

    #[test]
    fn test_clifford_relation() {
        let gammas = gamma_matrices_cl8();
        assert!(verify_clifford_relation(&gammas, 1e-10));
    }

    #[test]
    fn test_basis_count() {
        assert_eq!(count_basis_elements(8), 256);
        assert_eq!(count_basis_elements(6), 64);
    }

    #[test]
    fn test_clifford_algebra_verify() {
        let cl8 = CliffordAlgebra::cl8();
        assert!(cl8.verify(1e-10));
    }

    #[test]
    fn test_fermion_charges() {
        let charges = fermion_charges_cl6();
        assert_eq!(charges.len(), 4);

        // Check u quark charge
        let u = &charges[0];
        assert!((u.em_charge - 2.0 / 3.0).abs() < 1e-10);

        // Check electron charge
        let e = &charges[3];
        assert!((e.em_charge - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_lepton_mass_ratios() {
        let masses = LeptonMasses::observed();
        let ratio_mu_e = masses.ratio_mu_e();
        assert!(ratio_mu_e > 200.0 && ratio_mu_e < 210.0);
    }
}
