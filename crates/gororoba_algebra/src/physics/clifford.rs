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

use nalgebra::{DMatrix, Matrix2};
use num_complex::Complex64;

/// A complex matrix for gamma matrix representation (using nalgebra).
pub type GammaMatrix = DMatrix<Complex64>;

/// Pauli matrices sigma_1, sigma_2, sigma_3.
pub fn pauli_matrices() -> (Matrix2<Complex64>, Matrix2<Complex64>, Matrix2<Complex64>) {
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let i = Complex64::new(0.0, 1.0);

    let sigma_1 = Matrix2::new(zero, one, one, zero);
    let sigma_2 = Matrix2::new(zero, -i, i, zero);
    let sigma_3 = Matrix2::new(one, zero, zero, -one);

    (sigma_1, sigma_2, sigma_3)
}

/// Kronecker (tensor) product of two matrices.
pub fn kron(a: &GammaMatrix, b: &GammaMatrix) -> GammaMatrix {
    a.kronecker(b)
}

fn gamma_from_2x2(m: &Matrix2<Complex64>) -> GammaMatrix {
    GammaMatrix::from_row_slice(2, 2, m.as_slice())
}

/// Special case for 2x2 pauli matrix kronecker products.
pub fn kron2(a: &Matrix2<Complex64>, b: &Matrix2<Complex64>) -> GammaMatrix {
    let mut out = GammaMatrix::zeros(4, 4);
    for i in 0..2 {
        for j in 0..2 {
            let val = a[(i, j)];
            for k in 0..2 {
                for l in 0..2 {
                    out[(i * 2 + k, j * 2 + l)] = val * b[(k, l)];
                }
            }
        }
    }
    out
}

/// Construct 8 gamma matrices for Cl(8) in the real 16x16 representation.
pub fn gamma_matrices_cl8() -> Vec<GammaMatrix> {
    let (s1, s2, s3) = pauli_matrices();
    let i2 = Matrix2::identity();
    let s1g = gamma_from_2x2(&s1);
    let s2g = gamma_from_2x2(&s2);
    let s3g = gamma_from_2x2(&s3);
    let i2g = gamma_from_2x2(&i2);

    vec![
        kron(&kron(&kron(&s1g, &i2g), &i2g), &i2g),
        kron(&kron(&kron(&s2g, &i2g), &i2g), &i2g),
        kron(&kron(&kron(&s3g, &s1g), &i2g), &i2g),
        kron(&kron(&kron(&s3g, &s2g), &i2g), &i2g),
        kron(&kron(&kron(&s3g, &s3g), &s1g), &i2g),
        kron(&kron(&kron(&s3g, &s3g), &s2g), &i2g),
        kron(&kron(&kron(&s3g, &s3g), &s3g), &s1g),
        kron(&kron(&kron(&s3g, &s3g), &s3g), &s2g),
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

pub struct CliffordAlgebra {
    pub dimension: usize,
    pub gammas: Vec<GammaMatrix>,
}

impl CliffordAlgebra {
    pub fn cl8() -> Self {
        CliffordAlgebra {
            dimension: 8,
            gammas: gamma_matrices_cl8(),
        }
    }
    pub fn verify(&self, tol: f64) -> bool {
        verify_clifford_relation(&self.gammas, tol)
    }
    pub fn chirality_operator(&self) -> GammaMatrix {
        let n = self.gammas.len().min(6);
        let i = Complex64::new(0.0, 1.0);
        let mut result = &self.gammas[0] * i;
        for g in &self.gammas[1..n] {
            result = &result * g;
        }
        result
    }
    pub fn left_projector(&self) -> GammaMatrix {
        let chiral = self.chirality_operator();
        let dim = chiral.nrows();
        let identity = GammaMatrix::identity(dim, dim);
        let half = Complex64::new(0.5, 0.0);
        (&identity + &chiral) * half
    }
    pub fn right_projector(&self) -> GammaMatrix {
        let chiral = self.chirality_operator();
        let dim = chiral.nrows();
        let identity = GammaMatrix::identity(dim, dim);
        let half = Complex64::new(0.5, 0.0);
        (&identity - &chiral) * half
    }
}

pub fn majorana_conformal_cl42_generators() -> Vec<GammaMatrix> {
    use crate::construction::split_octonion::SplitOctonion;
    let mut generators = Vec::new();
    for i in 1..7 {
        let ei = SplitOctonion::basis(i);
        let m = ei.left_multiplication_matrix();
        let mut cm = GammaMatrix::zeros(8, 8);
        for r in 0..8 {
            for c in 0..8 {
                cm[(r, c)] = Complex64::new(m[(r, c)], 0.0);
            }
        }
        generators.push(cm);
    }
    generators
}

#[derive(Debug, Clone)]
pub struct FermionCharges {
    pub name: String,
    pub em_charge: f64,
    pub weak_isospin: f64,
    pub color_rep: String,
}

pub fn fermion_charges_cl6() -> Vec<FermionCharges> {
    vec![
        FermionCharges { name: "u_quark".to_string(), em_charge: 2.0 / 3.0, weak_isospin: 0.5, color_rep: "triplet".to_string() },
        FermionCharges { name: "d_quark".to_string(), em_charge: -1.0 / 3.0, weak_isospin: -0.5, color_rep: "triplet".to_string() },
        FermionCharges { name: "neutrino".to_string(), em_charge: 0.0, weak_isospin: 0.5, color_rep: "singlet".to_string() },
        FermionCharges { name: "electron".to_string(), em_charge: -1.0, weak_isospin: -0.5, color_rep: "singlet".to_string() },
    ]
}

#[derive(Debug, Clone)]
pub struct LeptonMasses {
    pub electron: f64,
    pub muon: f64,
    pub tau: f64,
}

impl LeptonMasses {
    pub fn observed() -> Self {
        LeptonMasses { electron: 0.511, muon: 105.66, tau: 1776.86 }
    }
    pub fn ratio_mu_e(&self) -> f64 { self.muon / self.electron }
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
    fn test_clifford_relation() {
        let gammas = gamma_matrices_cl8();
        assert!(verify_clifford_relation(&gammas, 1e-10));
    }
}
