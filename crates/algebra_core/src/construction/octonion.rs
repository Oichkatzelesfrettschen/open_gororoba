//! Octonion algebra: the unique 8-dimensional normed composition algebra.
//!
//! Octonions O form the final level in the Cayley-Dickson construction hierarchy:
//! R (1D) → C (2D) → H (4D) → O (8D).
//!
//! Key properties:
//! - Normed division algebra: N(xy) = N(x)N(y) where N(x) = ||x||²
//! - Non-associative but alternative: satisfies Moufang identities
//! - Basis: {1, e1, e2, ..., e7} with multiplication governed by Fano plane mnemonic
//! - Automorphism group: G2 (14-dimensional exceptional Lie group)
//! - Composition algebra: unique by Hurwitz' theorem for dim=8
//!
//! Reference: Baez "The Octonions" (2002), arXiv:0902.0431 (Yokota, 2009)

use std::fmt;

/// Octonion: 8-dimensional composition algebra.
/// Represented as 8 real components: [c0, c1, c2, ..., c7]
/// where c0 is scalar part and c1-c7 are imaginary parts {e1, e2, ..., e7}.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Octonion {
    /// 8 real components
    pub components: [f64; 8],
}

impl Octonion {
    /// Create octonion from 8 real components: (c0 + c1*e1 + ... + c7*e7)
    pub fn new(c: [f64; 8]) -> Self {
        Octonion { components: c }
    }

    /// Real unit: 1 + 0*e_i
    pub fn unit() -> Self {
        Octonion {
            components: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    }

    /// Zero: 0 + 0*e_i
    pub fn zero() -> Self {
        Octonion {
            components: [0.0; 8],
        }
    }

    /// Basis element e_i where i ∈ {0..7}.
    /// e_0 = 1 (scalar), e_1...e_7 = imaginary units.
    pub fn basis(i: usize) -> Self {
        assert!(i < 8, "Basis index must be 0-7");
        let mut c = [0.0; 8];
        c[i] = 1.0;
        Octonion { components: c }
    }

    /// Scalar part (c0)
    pub fn scalar(&self) -> f64 {
        self.components[0]
    }

    /// Imaginary part (c1*e1 + ... + c7*e7) as slice
    pub fn imaginary(&self) -> &[f64] {
        &self.components[1..]
    }

    /// Conjugate: c0 - c1*e1 - ... - c7*e7
    pub fn conjugate(&self) -> Octonion {
        let mut conj = self.components;
        for i in 1..8 {
            conj[i] = -conj[i];
        }
        Octonion { components: conj }
    }

    /// Norm squared: N(x) = x*conj(x) = c0² + c1² + ... + c7²
    pub fn norm_squared(&self) -> f64 {
        self.components.iter().map(|c| c * c).sum()
    }

    /// Euclidean norm: sqrt(N(x))
    pub fn norm(&self) -> f64 {
        self.norm_squared().sqrt()
    }

    /// Trace: T(x) = x + conj(x) = 2*c0
    pub fn trace(&self) -> f64 {
        2.0 * self.components[0]
    }

    /// Multiplicative inverse: x^(-1) = conj(x) / N(x)
    /// Panics if N(x) = 0.
    pub fn inverse(&self) -> Octonion {
        let n_sq = self.norm_squared();
        assert!(n_sq > 1e-15, "Cannot invert zero octonion");
        let conj = self.conjugate();
        Octonion {
            components: conj.components.map(|c| c / n_sq),
        }
    }

    /// Composition property: N(xy) = N(x)N(y).
    /// Verify by computing actual norm of product vs product of norms.
    pub fn verify_composition_property(&self, other: &Octonion, tolerance: f64) -> bool {
        let product = self.multiply(other);
        let lhs = product.norm_squared();
        let rhs = self.norm_squared() * other.norm_squared();
        (lhs - rhs).abs() < tolerance
    }

    /// Fano plane mnemonic multiplication for octonion basis elements.
    /// Returns the Fano plane multiplication rule e_i * e_j.
    /// Fano plane: 7 points, 7 lines, encodes octonion multiplication.
    /// Structure: three orthogonal complex planes (e1,e2,e3), (e4,e5,e6), (e0=1,e7)
    /// with cyclic identities and sign patterns.
    fn fano_multiply(i: usize, j: usize) -> (usize, i32) {
        // (i, j) -> (basis_index, sign)
        // Fano plane encodes all non-trivial products via 480 octonion algebras
        // Using standard representation: multiplication table computed from Fano lines
        match (i, j) {
            // Scalar multiplication (e0 = 1)
            (0, j) => (j, 1),
            (i, 0) => (i, 1),
            // Diagonal elements: e_i * e_i = -1 for i >= 1
            (1, 1) | (2, 2) | (3, 3) | (4, 4) | (5, 5) | (6, 6) | (7, 7) => (0, -1),
            // Fano line 1: (e1, e2, e3) cyclic with sign
            (1, 2) => (3, 1),   // e1*e2 = e3
            (2, 1) => (3, -1),  // e2*e1 = -e3
            (2, 3) => (1, 1),   // e2*e3 = e1
            (3, 2) => (1, -1),  // e3*e2 = -e1
            (3, 1) => (2, 1),   // e3*e1 = e2
            (1, 3) => (2, -1),  // e1*e3 = -e2
            // Fano line 2: (e4, e5, e6) cyclic with sign
            (4, 5) => (6, 1),   // e4*e5 = e6
            (5, 4) => (6, -1),  // e5*e4 = -e6
            (5, 6) => (4, 1),   // e5*e6 = e4
            (6, 5) => (4, -1),  // e6*e5 = -e4
            (6, 4) => (5, 1),   // e6*e4 = e5
            (4, 6) => (5, -1),  // e4*e6 = -e5
            // Fano line 3: (e7, e1, e4) and cyclic
            (7, 1) => (4, 1),   // e7*e1 = e4
            (1, 7) => (4, -1),  // e1*e7 = -e4
            (1, 4) => (7, 1),   // e1*e4 = e7
            (4, 1) => (7, -1),  // e4*e1 = -e7
            (4, 7) => (1, 1),   // e4*e7 = e1
            (7, 4) => (1, -1),  // e7*e4 = -e1
            // Fano line 4: (e7, e2, e5)
            (7, 2) => (5, 1),   // e7*e2 = e5
            (2, 7) => (5, -1),  // e2*e7 = -e5
            (2, 5) => (7, 1),   // e2*e5 = e7
            (5, 2) => (7, -1),  // e5*e2 = -e7
            (5, 7) => (2, 1),   // e5*e7 = e2
            (7, 5) => (2, -1),  // e7*e5 = -e2
            // Fano line 5: (e7, e3, e6)
            (7, 3) => (6, 1),   // e7*e3 = e6
            (3, 7) => (6, -1),  // e3*e7 = -e6
            (3, 6) => (7, 1),   // e3*e6 = e7
            (6, 3) => (7, -1),  // e6*e3 = -e7
            (6, 7) => (3, 1),   // e6*e7 = e3
            (7, 6) => (3, -1),  // e7*e6 = -e3
            // Remaining cross products (derived from Fano plane structure)
            (1, 5) => (6, -1),  // e1*e5 = -e6
            (5, 1) => (6, 1),   // e5*e1 = e6
            (2, 4) => (6, 1),   // e2*e4 = e6
            (4, 2) => (6, -1),  // e4*e2 = -e6
            (3, 5) => (7, -1),  // e3*e5 = -e7
            (5, 3) => (7, 1),   // e5*e3 = e7
            (1, 6) => (5, 1),   // e1*e6 = e5
            (6, 1) => (5, -1),  // e6*e1 = -e5
            (2, 6) => (4, -1),  // e2*e6 = -e4
            (6, 2) => (4, 1),   // e6*e2 = e4
            (3, 4) => (5, -1),  // e3*e4 = -e5
            (4, 3) => (5, 1),   // e4*e3 = e5
            _ => (0, 0),
        }
    }

    /// Octonion multiplication: (c0 + Σc_i*e_i) * (d0 + Σd_j*e_j)
    /// Uses Fano plane mnemonic to encode the multiplication table.
    pub fn multiply(&self, other: &Octonion) -> Octonion {
        let mut result = [0.0; 8];

        // Expand multiplication using basis element products
        for i in 0..8 {
            for j in 0..8 {
                let ci = self.components[i];
                let dj = other.components[j];
                if ci.abs() < 1e-15 || dj.abs() < 1e-15 {
                    continue;
                }

                let (prod_basis, sign) = Octonion::fano_multiply(i, j);
                result[prod_basis] += sign as f64 * ci * dj;
            }
        }

        Octonion { components: result }
    }

    /// Check if self satisfies Moufang identity: a(x(ay)) = (axa)y
    pub fn moufang_identity_left(&self, x: &Octonion, y: &Octonion, tolerance: f64) -> bool {
        let ay = self.multiply(y);
        let xay = x.multiply(&ay);
        let lhs = self.multiply(&xay);

        let axa = self.multiply(self);
        let rhs = axa.multiply(x).multiply(y);

        let diff: f64 = lhs
            .components
            .iter()
            .zip(rhs.components.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        diff < tolerance
    }

    /// Check if self satisfies Moufang identity: ((xa)y)a = x(aya)
    pub fn moufang_identity_right(&self, x: &Octonion, y: &Octonion, tolerance: f64) -> bool {
        let xa = x.multiply(self);
        let xay = xa.multiply(y);
        let lhs = xay.multiply(self);

        let ay = y.multiply(self);
        let aya = ay.multiply(self);
        let rhs = x.multiply(&aya);

        let diff: f64 = lhs
            .components
            .iter()
            .zip(rhs.components.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        diff < tolerance
    }

    /// Flexibility identity: (xy)x = x(yx)
    pub fn flexibility_identity(&self, y: &Octonion, tolerance: f64) -> bool {
        let lhs = self.multiply(y).multiply(self);
        let rhs = self.multiply(y.multiply(self));
        let diff: f64 = lhs
            .components
            .iter()
            .zip(rhs.components.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        diff < tolerance
    }

    /// Associator: [x, y, z] = (xy)z - x(yz).
    /// For octonions, this is nonzero (non-associative).
    pub fn associator(&self, y: &Octonion, z: &Octonion) -> Octonion {
        let lhs = self.multiply(y).multiply(z);
        let rhs = self.multiply(&y.multiply(z));
        Octonion {
            components: std::array::from_fn(|i| lhs.components[i] - rhs.components[i]),
        }
    }
}

impl fmt::Display for Octonion {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{:.4} + {:.4}e1 + {:.4}e2 + {:.4}e3 + {:.4}e4 + {:.4}e5 + {:.4}e6 + {:.4}e7",
            self.components[0],
            self.components[1],
            self.components[2],
            self.components[3],
            self.components[4],
            self.components[5],
            self.components[6],
            self.components[7]
        )
    }
}

impl Default for Octonion {
    fn default() -> Self {
        Octonion::unit()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_octonion_basis() {
        let e0 = Octonion::basis(0);
        assert_eq!(e0.scalar(), 1.0);
        assert_eq!(e0.norm(), 1.0);

        for i in 1..8 {
            let ei = Octonion::basis(i);
            assert_eq!(ei.components[i], 1.0);
            assert_eq!(ei.norm(), 1.0);
        }
    }

    #[test]
    fn test_octonion_conjugate() {
        let oct = Octonion::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let conj = oct.conjugate();
        assert_eq!(conj.components[0], 1.0);
        for i in 1..8 {
            assert_eq!(conj.components[i], -oct.components[i]);
        }
    }

    #[test]
    fn test_octonion_norm() {
        let oct = Octonion::new([3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let n_sq = oct.norm_squared();
        assert!((n_sq - 25.0).abs() < 1e-10); // 3² + 4² = 25
        assert!((oct.norm() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_octonion_composition_property() {
        let x = Octonion::new([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let y = Octonion::new([1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert!(x.verify_composition_property(&y, 1e-10));
    }

    #[test]
    fn test_octonion_basis_multiply_diagonal() {
        // e_i * e_i = -1 for i >= 1
        for i in 1..8 {
            let ei = Octonion::basis(i);
            let product = ei.multiply(&ei);
            assert!((product.components[0] - (-1.0)).abs() < 1e-10);
            for j in 1..8 {
                assert!(product.components[j].abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_octonion_fano_line_1_2_3() {
        // e1*e2 = e3
        let e1 = Octonion::basis(1);
        let e2 = Octonion::basis(2);
        let _e3 = Octonion::basis(3);
        let prod = e1.multiply(&e2);
        assert!((prod.components[3] - 1.0).abs() < 1e-10);
        for i in 0..8 {
            if i != 3 {
                assert!(prod.components[i].abs() < 1e-10);
            }
        }

        // e2*e1 = -e3
        let prod_rev = e2.multiply(&e1);
        assert!((prod_rev.components[3] - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_octonion_inverse() {
        let x = Octonion::new([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let x_inv = x.inverse();
        let product = x.multiply(&x_inv);
        // Should be close to 1
        assert!((product.components[0] - 1.0).abs() < 1e-10);
        for i in 1..8 {
            assert!(product.components[i].abs() < 1e-10);
        }
    }

    #[test]
    fn test_octonion_associator_nonzero() {
        let e1 = Octonion::basis(1);
        let e2 = Octonion::basis(2);
        let e4 = Octonion::basis(4);
        let assoc = e1.associator(&e2, &e4);
        // Associator should be nonzero for most octonion triples
        let norm = assoc.norm();
        assert!(norm > 1e-10, "Associator should be nonzero for generic octonions");
    }

    #[test]
    fn test_octonion_trace() {
        let oct = Octonion::new([5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        let tr = oct.trace();
        assert!((tr - 10.0).abs() < 1e-10); // T(x) = 2*c0 = 10
    }

    #[test]
    fn test_octonion_flexibility() {
        let x = Octonion::new([1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let y = Octonion::new([1.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0]);
        // Flexibility identity holds for all alternative algebras
        assert!(x.flexibility_identity(&y, 1e-10));
    }

    #[test]
    fn test_octonion_unit_scalar_product() {
        // 1 * x = x * 1 = x
        let x = Octonion::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let one = Octonion::unit();
        let prod1 = one.multiply(&x);
        let prod2 = x.multiply(&one);
        for i in 0..8 {
            assert!((prod1.components[i] - x.components[i]).abs() < 1e-10);
            assert!((prod2.components[i] - x.components[i]).abs() < 1e-10);
        }
    }
}
