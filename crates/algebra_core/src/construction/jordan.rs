// Phase 10.1: Jordan Algebras Research
// Exceptional algebras family: A_1 (reals), A_2 (3x3 symmetric matrices)
// Construction: symmetrized product a o b = (ab + ba) / 2
// Cross-validation: commutativity vs Phase 9 (tessarines 100%, CD 0%)
// Date: 2026-02-10, Sprint 35-36

use nalgebra::Matrix3;

/// Jordan algebra A_1: Real numbers with symmetrized product
#[derive(Clone, Copy, Debug)]
pub struct JordanA1(pub f64);

impl JordanA1 {
    pub fn new(x: f64) -> Self {
        JordanA1(x)
    }
    pub fn jordan_product(self, other: JordanA1) -> JordanA1 {
        JordanA1((self.0 * other.0 + other.0 * self.0) / 2.0)
    }
    pub fn is_commutative_with(self, other: JordanA1) -> bool {
        let prod1 = self.jordan_product(other);
        let prod2 = other.jordan_product(self);
        (prod1.0 - prod2.0).abs() < 1e-14
    }
    pub fn satisfies_jordan_identity(self, other: JordanA1) -> bool {
        let a_sq = self.jordan_product(self);
        let ab = self.jordan_product(other);
        let lhs = ab.jordan_product(a_sq);
        let b_aa = other.jordan_product(a_sq);
        let rhs = self.jordan_product(b_aa);
        (lhs.0 - rhs.0).abs() < 1e-14
    }
}

/// Jordan algebra A_2: 3x3 symmetric matrices with symmetrized product
#[derive(Clone, Copy, Debug)]
pub struct JordanA2 {
    data: [f64; 6], // [a11, a22, a33, a12, a13, a23]
}

impl JordanA2 {
    pub fn identity() -> Self {
        JordanA2 {
            data: [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        }
    }
    pub fn new(a11: f64, a22: f64, a33: f64, a12: f64, a13: f64, a23: f64) -> Self {
        JordanA2 {
            data: [a11, a22, a33, a12, a13, a23],
        }
    }
    fn to_matrix3(self) -> Matrix3<f64> {
        let [a11, a22, a33, a12, a13, a23] = self.data;
        Matrix3::new(a11, a12, a13, a12, a22, a23, a13, a23, a33)
    }
    pub fn jordan_product(self, other: JordanA2) -> JordanA2 {
        let m1 = self.to_matrix3();
        let m2 = other.to_matrix3();
        let prod = (m1 * m2 + m2 * m1) / 2.0;
        let [a11, a22, a33, a12, a13, a23] = [
            prod[(0, 0)],
            prod[(1, 1)],
            prod[(2, 2)],
            prod[(0, 1)],
            prod[(0, 2)],
            prod[(1, 2)],
        ];
        JordanA2::new(a11, a22, a33, a12, a13, a23)
    }
    pub fn is_commutative_with(self, other: JordanA2) -> bool {
        let prod1 = self.jordan_product(other);
        let prod2 = other.jordan_product(self);
        prod1
            .data
            .iter()
            .zip(prod2.data.iter())
            .all(|(x, y)| (x - y).abs() < 1e-12)
    }
    pub fn satisfies_jordan_identity(self, other: JordanA2) -> bool {
        let a_sq = self.jordan_product(self);
        let ab = self.jordan_product(other);
        let lhs = ab.jordan_product(a_sq);
        let b_aa = other.jordan_product(a_sq);
        let rhs = self.jordan_product(b_aa);
        lhs.data
            .iter()
            .zip(rhs.data.iter())
            .all(|(x, y)| (x - y).abs() < 1e-12)
    }
    pub fn determinant(self) -> f64 {
        self.to_matrix3().determinant()
    }
    pub fn is_invertible(self) -> bool {
        self.determinant().abs() > 1e-14
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jordan_a1_commutativity() {
        let a = JordanA1::new(2.5);
        let b = JordanA1::new(3.7);
        assert!(a.is_commutative_with(b));
    }

    #[test]
    fn test_jordan_a1_jordan_identity() {
        let a = JordanA1::new(2.0);
        let b = JordanA1::new(3.5);
        assert!(a.satisfies_jordan_identity(b));
    }

    #[test]
    fn test_jordan_a2_commutativity() {
        let a = JordanA2::new(1.0, 2.0, 3.0, 0.5, 0.3, 0.7);
        let b = JordanA2::new(2.0, 1.5, 2.5, 0.8, 0.4, 0.6);
        assert!(a.is_commutative_with(b));
    }

    #[test]
    fn test_jordan_a2_jordan_identity() {
        let a = JordanA2::new(1.0, 2.0, 3.0, 0.5, 0.3, 0.7);
        let b = JordanA2::new(2.0, 1.5, 2.5, 0.8, 0.4, 0.6);
        assert!(a.satisfies_jordan_identity(b));
    }

    #[test]
    fn test_jordan_algebras_100_percent_commutative() {
        // Cross-validation: Jordan algebras are ALWAYS commutative
        // Contrasts with CD dim>=4 (0% commutativity)
        let samples_a1 = [JordanA1::new(1.0), JordanA1::new(2.0), JordanA1::new(3.0)];
        let samples_a2 = [
            JordanA2::new(1.0, 2.0, 3.0, 0.5, 0.3, 0.7),
            JordanA2::new(2.0, 1.5, 2.5, 0.8, 0.4, 0.6),
            JordanA2::new(0.5, 0.8, 1.2, 0.2, 0.1, 0.3),
        ];

        for i in 0..samples_a1.len() {
            for j in 0..samples_a1.len() {
                assert!(samples_a1[i].is_commutative_with(samples_a1[j]));
            }
        }
        for i in 0..samples_a2.len() {
            for j in 0..samples_a2.len() {
                assert!(samples_a2[i].is_commutative_with(samples_a2[j]));
            }
        }
    }
}
