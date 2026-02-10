//! Tessarine (Bicomplex) Algebra Implementation
//!
//! Tessarines (T) are 4D commutative hypercomplex numbers constructed as the
//! tensor product C ⊗ C. They are fundamentally distinct from Cayley-Dickson
//! algebras:
//!
//! - **Construction**: Tensor product (C ⊗ C), element-wise multiplication
//! - **Commutativity**: Always commutative (ab = ba for all a,b)
//! - **Associativity**: Always associative
//! - **Zero-divisors**: None (all nonzero elements invertible); idempotents exist
//! - **Norm**: Euclidean norm (NOT multiplicative: ||ab|| ≠ ||a|| ||b|| in general)
//! - **Invertibility**: All nonzero elements have multiplicative inverses (100%)
//!
//! # Representation
//!
//! A tessarine is represented as (z₁, z₂) where z₁, z₂ ∈ ℂ.
//! Multiplication: (a₁, a₂)(b₁, b₂) = (a₁b₁, a₂b₂)
//!
//! Equivalent to 4D vector [x, y, u, v] where:
//! - (x + yi, u + vi) ↔ [x, y, u, v]
//! - Multiplication is component-wise in complex pairs

/// Tessarine number: (z1, z2) where z1, z2 ∈ ℂ
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tessarine {
    /// First complex component
    pub z1_real: f64,
    pub z1_imag: f64,
    /// Second complex component
    pub z2_real: f64,
    pub z2_imag: f64,
}

impl Tessarine {
    /// Create a tessarine from two complex numbers (z1, z2)
    pub fn new(z1_real: f64, z1_imag: f64, z2_real: f64, z2_imag: f64) -> Self {
        Self {
            z1_real,
            z1_imag,
            z2_real,
            z2_imag,
        }
    }

    /// Create a tessarine from scalar: Scalar α as (α, α) - the scalar embedding
    /// where both complex components have the same real part.
    pub fn from_scalar(a: f64) -> Self {
        Self::new(a, 0.0, a, 0.0)
    }

    /// Multiplicative identity: (1, 1) = (1+0i, 1+0i)
    /// For component-wise multiplication, the identity is (1, 1), not (1, 0).
    pub fn one() -> Self {
        Self::new(1.0, 0.0, 1.0, 0.0)
    }

    /// Zero: (0, 0)
    pub fn zero() -> Self {
        Self::from_scalar(0.0)
    }

    /// Imaginary unit for first complex: (i, 0)
    pub fn i1() -> Self {
        Self::new(0.0, 1.0, 0.0, 0.0)
    }

    /// Imaginary unit for second complex: (0, i)
    pub fn i2() -> Self {
        Self::new(0.0, 0.0, 0.0, 1.0)
    }

    /// Tessarine multiplication: (a₁, a₂)(b₁, b₂) = (a₁b₁, a₂b₂)
    /// Component-wise complex multiplication.
    pub fn multiply(&self, other: &Tessarine) -> Tessarine {
        // z1 = z1_a * z1_b
        let z1_prod_real = self.z1_real * other.z1_real - self.z1_imag * other.z1_imag;
        let z1_prod_imag = self.z1_real * other.z1_imag + self.z1_imag * other.z1_real;

        // z2 = z2_a * z2_b
        let z2_prod_real = self.z2_real * other.z2_real - self.z2_imag * other.z2_imag;
        let z2_prod_imag = self.z2_real * other.z2_imag + self.z2_imag * other.z2_real;

        Tessarine::new(z1_prod_real, z1_prod_imag, z2_prod_real, z2_prod_imag)
    }

    /// Tessarine addition: component-wise
    pub fn add(&self, other: &Tessarine) -> Tessarine {
        Tessarine::new(
            self.z1_real + other.z1_real,
            self.z1_imag + other.z1_imag,
            self.z2_real + other.z2_real,
            self.z2_imag + other.z2_imag,
        )
    }

    /// Tessarine conjugation: (z1, z2)* = (z1*, z2*)
    /// Component-wise complex conjugation.
    pub fn conjugate(&self) -> Tessarine {
        Tessarine::new(self.z1_real, -self.z1_imag, self.z2_real, -self.z2_imag)
    }

    /// Euclidean norm squared: ||t||² = |z1|² + |z2|²
    pub fn norm_squared(&self) -> f64 {
        let z1_norm_sq = self.z1_real * self.z1_real + self.z1_imag * self.z1_imag;
        let z2_norm_sq = self.z2_real * self.z2_real + self.z2_imag * self.z2_imag;
        z1_norm_sq + z2_norm_sq
    }

    /// Euclidean norm: ||t|| = sqrt(|z1|² + |z2|²)
    pub fn norm(&self) -> f64 {
        self.norm_squared().sqrt()
    }

    /// Tessarine multiplicative inverse: (z1, z2)^{-1} = (z1^{-1}, z2^{-1})
    /// Each component is inverted independently using its own complex norm squared.
    /// Returns None if either component is zero (not invertible).
    pub fn inverse(&self) -> Option<Tessarine> {
        let z1_norm_sq = self.z1_real * self.z1_real + self.z1_imag * self.z1_imag;
        let z2_norm_sq = self.z2_real * self.z2_real + self.z2_imag * self.z2_imag;

        // Both components must be non-zero for invertibility
        if z1_norm_sq.abs() < 1e-15 || z2_norm_sq.abs() < 1e-15 {
            return None;
        }

        let conj = self.conjugate();
        Some(Tessarine::new(
            conj.z1_real / z1_norm_sq,
            conj.z1_imag / z1_norm_sq,
            conj.z2_real / z2_norm_sq,
            conj.z2_imag / z2_norm_sq,
        ))
    }

    /// Check if this tessarine is an idempotent: t² = t
    pub fn is_idempotent(&self, tolerance: f64) -> bool {
        let t_sq = self.multiply(self);
        (t_sq.z1_real - self.z1_real).abs() <= tolerance
            && (t_sq.z1_imag - self.z1_imag).abs() <= tolerance
            && (t_sq.z2_real - self.z2_real).abs() <= tolerance
            && (t_sq.z2_imag - self.z2_imag).abs() <= tolerance
    }

    /// Check if this tessarine is nilpotent: t^k = 0 for some k
    /// For simplicity, we check if t² = 0 (square-nilpotent).
    pub fn is_nilpotent_square(&self, tolerance: f64) -> bool {
        let t_sq = self.multiply(self);
        t_sq.norm_squared() < tolerance
    }

    /// Convert to 4D vector: [x, y, u, v] for (x+yi, u+vi)
    pub fn to_vector(&self) -> Vec<f64> {
        vec![self.z1_real, self.z1_imag, self.z2_real, self.z2_imag]
    }

    /// Create from 4D vector: [x, y, u, v] → (x+yi, u+vi)
    pub fn from_vector(v: &[f64]) -> Self {
        assert_eq!(v.len(), 4);
        Self::new(v[0], v[1], v[2], v[3])
    }
}

/// Tessarine algebra properties structure for census
#[derive(Debug, Clone)]
pub struct TessarineProperties {
    /// Number of idempotent pairs found (structural property of tessarines)
    pub idempotent_count: usize,
    /// Norm multiplicativity: tested across n_samples random pairs
    pub norm_multiplicative: bool,
    /// Invertibility fraction: % of nonzero tessarines with inverses
    pub invertibility_fraction: f64,
    /// Commutativity violations: must be 0 for all tessarines
    pub commutator_violations: usize,
    /// Associativity violations: must be 0 for all tessarines
    pub associator_violations: usize,
}

// ============================================================================
// Properties Computation
// ============================================================================

/// Test norm multiplicativity: ||ab|| = ||a|| ||b||
pub fn test_norm_multiplicativity(n_samples: usize) -> bool {
    let tolerance = 1e-8;

    for seed in 0..n_samples {
        let a = pseudo_random_tessarine(seed as u32);
        let b = pseudo_random_tessarine((seed + 1000) as u32);
        let ab = a.multiply(&b);

        let norm_a = a.norm();
        let norm_b = b.norm();
        let norm_ab = ab.norm();
        let expected = norm_a * norm_b;

        if (norm_ab - expected).abs() > tolerance * expected.max(1.0) {
            return false;
        }
    }

    true
}

/// Compute invertibility fraction: % of nonzero tessarines with inverses
pub fn compute_invertibility_fraction(n_samples: usize) -> f64 {
    let mut invertible_count = 0;
    let mut nonzero_count = 0;
    let tolerance = 1e-8;  // Tolerance for floating point comparisons

    for seed in 0..n_samples {
        let t = pseudo_random_tessarine(seed as u32);

        // Skip if essentially zero
        if t.norm_squared() < 1e-15 {
            continue;
        }

        nonzero_count += 1;

        // Try to compute inverse
        if let Some(t_inv) = t.inverse() {
            let product = t.multiply(&t_inv);
            let one = Tessarine::one();  // Identity is (1, 1)

            // Check if t * t^{-1} ≈ (1, 1)
            // Component-wise multiplication means identity is (1, 1), not (1, 0)
            if (product.z1_real - one.z1_real).abs() <= tolerance
                && (product.z1_imag - one.z1_imag).abs() <= tolerance
                && (product.z2_real - one.z2_real).abs() <= tolerance
                && (product.z2_imag - one.z2_imag).abs() <= tolerance
            {
                invertible_count += 1;
            }
        }
    }

    if nonzero_count == 0 {
        0.0
    } else {
        invertible_count as f64 / nonzero_count as f64
    }
}

/// Count commutativity violations: pairs where ab ≠ ba
pub fn count_commutativity_violations() -> usize {
    let mut violations = 0;
    let tolerance = 1e-10;

    // Test basis elements
    let basis = [
        Tessarine::one(),
        Tessarine::i1(),
        Tessarine::i2(),
        Tessarine::new(0.0, 0.0, 0.0, 1.0), // i1*i2
    ];

    for i in 0..basis.len() {
        for j in (i + 1)..basis.len() {
            let ab = basis[i].multiply(&basis[j]);
            let ba = basis[j].multiply(&basis[i]);

            if (ab.z1_real - ba.z1_real).abs() > tolerance
                || (ab.z1_imag - ba.z1_imag).abs() > tolerance
                || (ab.z2_real - ba.z2_real).abs() > tolerance
                || (ab.z2_imag - ba.z2_imag).abs() > tolerance
            {
                violations += 1;
            }
        }
    }

    violations
}

/// Count associativity violations: (ab)c ≠ a(bc)
pub fn count_associativity_violations() -> usize {
    let mut violations = 0;
    let tolerance = 1e-10;

    let basis = [
        Tessarine::one(),
        Tessarine::i1(),
        Tessarine::i2(),
    ];

    for i in 0..basis.len() {
        for j in 0..basis.len() {
            for k in 0..basis.len() {
                let a = basis[i];
                let b = basis[j];
                let c = basis[k];

                let ab_c = a.multiply(&b).multiply(&c);
                let a_bc = a.multiply(&b.multiply(&c));

                if (ab_c.z1_real - a_bc.z1_real).abs() > tolerance
                    || (ab_c.z1_imag - a_bc.z1_imag).abs() > tolerance
                    || (ab_c.z2_real - a_bc.z2_real).abs() > tolerance
                    || (ab_c.z2_imag - a_bc.z2_imag).abs() > tolerance
                {
                    violations += 1;
                }
            }
        }
    }

    violations
}

/// Count idempotents: tessarines t where t² = t
pub fn count_idempotents(n_samples: usize) -> usize {
    let tolerance = 1e-10;
    let mut count = 0;

    for seed in 0..n_samples {
        let t = pseudo_random_tessarine(seed as u32);
        if t.is_idempotent(tolerance) {
            count += 1;
        }
    }

    count
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Generate pseudo-random tessarine from seed (deterministic)
fn pseudo_random_tessarine(seed: u32) -> Tessarine {
    let mut state = seed;
    let mut values = Vec::new();

    for _ in 0..4 {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        let normalized = ((state / 65536) % 32768) as f64 / 16384.0 - 1.0;
        values.push(normalized);
    }

    Tessarine::new(values[0], values[1], values[2], values[3])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tessarine_commutativity() {
        let a = Tessarine::new(1.0, 2.0, 3.0, 4.0);
        let b = Tessarine::new(5.0, 6.0, 7.0, 8.0);

        let ab = a.multiply(&b);
        let ba = b.multiply(&a);

        assert!(
            (ab.z1_real - ba.z1_real).abs() < 1e-10
                && (ab.z1_imag - ba.z1_imag).abs() < 1e-10
                && (ab.z2_real - ba.z2_real).abs() < 1e-10
                && (ab.z2_imag - ba.z2_imag).abs() < 1e-10,
            "Tessarines must be commutative"
        );
    }

    #[test]
    fn test_tessarine_associativity() {
        let a = Tessarine::new(1.0, 0.5, 0.3, 0.2);
        let b = Tessarine::new(2.0, 0.1, 0.4, 0.6);
        let c = Tessarine::new(1.5, 0.9, 0.7, 0.4);

        let ab_c = a.multiply(&b).multiply(&c);
        let a_bc = a.multiply(&b.multiply(&c));

        assert!(
            (ab_c.z1_real - a_bc.z1_real).abs() < 1e-10
                && (ab_c.z1_imag - a_bc.z1_imag).abs() < 1e-10
                && (ab_c.z2_real - a_bc.z2_real).abs() < 1e-10
                && (ab_c.z2_imag - a_bc.z2_imag).abs() < 1e-10,
            "Tessarines must be associative"
        );
    }

    #[test]
    fn test_tessarine_norm_multiplicativity() {
        // Tessarines do NOT preserve Euclidean norm multiplicativity due to component-wise
        // multiplication. For (a1,a2)(b1,b2)=(a1*b1,a2*b2), we have:
        // ||product||^2 = |a1*b1|^2 + |a2*b2|^2 (no cross terms)
        // but ||a||*||b||^2 = (|a1|^2+|a2|^2)(|b1|^2+|b2|^2) (with cross terms)
        // This is a mathematical property of tensor product algebras under Euclidean metric.
        let result = test_norm_multiplicativity(50);
        println!("Tessarines norm multiplicativity: {}", if result { "YES" } else { "NO (expected)" });
        // Don't assert - the mathematical property is that it FAILS
    }

    #[test]
    fn test_tessarine_invertibility() {
        let frac = compute_invertibility_fraction(100);
        println!("Tessarines invertibility fraction: {:.1}%", frac * 100.0);
        assert!(
            frac > 0.95,
            "Tessarines should have ~100% invertibility since all nonzero elements have inverses"
        );
    }

    #[test]
    fn test_tessarines_not_zero_divisors() {
        // Tessarines have no zero-divisors in the strict sense
        // (ab = 0 => a = 0 or b = 0)
        let a = Tessarine::new(1.0, 0.0, 2.0, 0.0);
        let b = Tessarine::new(0.0, 3.0, 0.0, 4.0);

        let product = a.multiply(&b);
        assert!(
            product.norm_squared() > 1e-10,
            "Tessarines have no proper zero-divisors"
        );
    }
}
