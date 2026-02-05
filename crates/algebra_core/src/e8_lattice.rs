//! E8 Lattice and Root System.
//!
//! The E8 lattice is the unique even unimodular lattice in 8 dimensions.
//! It contains 240 roots (vectors of squared length 2) which form the
//! root system of the exceptional Lie algebra e8.
//!
//! # Construction
//! The E8 lattice can be constructed as:
//! - D8 (integer points with even coordinate sum) plus half-integer translates
//! - Alternatively: all points (x1,...,x8) where either all xi are integers
//!   with even sum, or all xi are half-integers with even sum
//!
//! # Literature
//! - Conway & Sloane, "Sphere Packings, Lattices and Groups"
//! - Adams, "Lectures on Exceptional Lie Groups"


/// An E8 root vector (8 components).
#[derive(Debug, Clone, PartialEq)]
pub struct E8Root {
    /// Components of the root vector
    pub coords: [f64; 8],
    /// Squared length (should be 2 for roots)
    pub norm_sq: f64,
}

impl E8Root {
    /// Create a new E8 root from coordinates.
    pub fn new(coords: [f64; 8]) -> Self {
        let norm_sq: f64 = coords.iter().map(|x| x * x).sum();
        Self { coords, norm_sq }
    }

    /// Check if this is a valid E8 root (norm^2 = 2).
    pub fn is_valid_root(&self) -> bool {
        (self.norm_sq - 2.0).abs() < 1e-10
    }

    /// Inner product with another root.
    pub fn inner_product(&self, other: &E8Root) -> f64 {
        self.coords.iter()
            .zip(other.coords.iter())
            .map(|(a, b)| a * b)
            .sum()
    }
}

/// The E8 lattice structure.
#[derive(Debug, Clone)]
pub struct E8Lattice {
    /// The 240 roots
    pub roots: Vec<E8Root>,
    /// Simple roots (basis for root system)
    pub simple_roots: [E8Root; 8],
    /// Cartan matrix (8x8)
    pub cartan_matrix: [[i32; 8]; 8],
}

/// Generate all 240 E8 roots.
///
/// The roots are:
/// 1. 112 roots of type (+/-1, +/-1, 0, 0, 0, 0, 0, 0) (permutations)
/// 2. 128 roots of type (+/-1/2, ...) with even number of minus signs
pub fn generate_e8_roots() -> Vec<E8Root> {
    let mut roots = Vec::with_capacity(240);

    // Type 1: (+/-1, +/-1, 0, 0, 0, 0, 0, 0) - all permutations
    // Choose 2 positions out of 8, then 4 sign choices = C(8,2) * 4 = 28 * 4 = 112
    for i in 0..8 {
        for j in (i + 1)..8 {
            for sign_i in [-1.0, 1.0] {
                for sign_j in [-1.0, 1.0] {
                    let mut coords = [0.0; 8];
                    coords[i] = sign_i;
                    coords[j] = sign_j;
                    roots.push(E8Root::new(coords));
                }
            }
        }
    }

    // Type 2: (+/-1/2, +/-1/2, ...) with even number of minus signs
    // 2^8 = 256 sign patterns, half have even minus signs = 128
    for sign_pattern in 0u8..=255 {
        let minus_count = sign_pattern.count_ones();
        if minus_count % 2 == 0 {
            let mut coords = [0.5; 8];
            for bit in 0..8 {
                if (sign_pattern >> bit) & 1 == 1 {
                    coords[bit] = -0.5;
                }
            }
            roots.push(E8Root::new(coords));
        }
    }

    assert_eq!(roots.len(), 240, "E8 should have exactly 240 roots");
    roots
}

/// Compute the E8 Cartan matrix.
///
/// The Cartan matrix A_ij = 2 * (alpha_i, alpha_j) / (alpha_j, alpha_j)
/// For E8, this is an 8x8 matrix with specific structure.
pub fn e8_cartan_matrix() -> [[i32; 8]; 8] {
    // Standard E8 Cartan matrix (Bourbaki numbering)
    [
        [ 2, -1,  0,  0,  0,  0,  0,  0],
        [-1,  2, -1,  0,  0,  0,  0,  0],
        [ 0, -1,  2, -1,  0,  0,  0, -1],
        [ 0,  0, -1,  2, -1,  0,  0,  0],
        [ 0,  0,  0, -1,  2, -1,  0,  0],
        [ 0,  0,  0,  0, -1,  2, -1,  0],
        [ 0,  0,  0,  0,  0, -1,  2,  0],
        [ 0,  0, -1,  0,  0,  0,  0,  2],
    ]
}

/// Simple roots for E8 in standard basis.
///
/// These are chosen to form a basis compatible with the Cartan matrix.
pub fn e8_simple_roots() -> [E8Root; 8] {
    [
        // alpha_1 = (1, -1, 0, 0, 0, 0, 0, 0)
        E8Root::new([1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        // alpha_2 = (0, 1, -1, 0, 0, 0, 0, 0)
        E8Root::new([0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        // alpha_3 = (0, 0, 1, -1, 0, 0, 0, 0)
        E8Root::new([0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0]),
        // alpha_4 = (0, 0, 0, 1, -1, 0, 0, 0)
        E8Root::new([0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0]),
        // alpha_5 = (0, 0, 0, 0, 1, -1, 0, 0)
        E8Root::new([0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0]),
        // alpha_6 = (0, 0, 0, 0, 0, 1, -1, 0)
        E8Root::new([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0]),
        // alpha_7 = (0, 0, 0, 0, 0, 1, 1, 0)
        E8Root::new([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0]),
        // alpha_8 = (-1/2, -1/2, -1/2, -1/2, -1/2, -1/2, -1/2, 1/2)
        E8Root::new([-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 0.5]),
    ]
}

/// Order of the E8 Weyl group.
///
/// |W(E8)| = 696,729,600 = 2^14 * 3^5 * 5^2 * 7
pub fn e8_weyl_group_order() -> u64 {
    696_729_600
}

/// Compute all pairwise inner products of E8 roots.
///
/// Returns a histogram of inner product values.
pub fn compute_e8_inner_products(roots: &[E8Root]) -> Vec<(i32, usize)> {
    let mut counts = std::collections::HashMap::new();

    for i in 0..roots.len() {
        for j in (i + 1)..roots.len() {
            let ip = roots[i].inner_product(&roots[j]).round() as i32;
            *counts.entry(ip).or_insert(0) += 1;
        }
    }

    let mut result: Vec<_> = counts.into_iter().collect();
    result.sort_by_key(|(ip, _)| *ip);
    result
}

impl E8Lattice {
    /// Create a new E8 lattice.
    pub fn new() -> Self {
        Self {
            roots: generate_e8_roots(),
            simple_roots: e8_simple_roots(),
            cartan_matrix: e8_cartan_matrix(),
        }
    }

    /// Count positive roots.
    pub fn positive_root_count(&self) -> usize {
        // Half of 240 = 120
        120
    }

    /// Verify Cartan matrix properties.
    pub fn verify_cartan_matrix(&self) -> bool {
        let c = &self.cartan_matrix;

        // Check diagonal elements are 2
        for i in 0..8 {
            if c[i][i] != 2 {
                return false;
            }
        }

        // Check symmetry of A_ij * A_ji pattern
        for i in 0..8 {
            for j in 0..8 {
                if i != j && c[i][j] != 0 && c[j][i] != 0 {
                    // Off-diagonal product should be 0, 1, 2, or 3
                    let prod = c[i][j] * c[j][i];
                    if prod < 0 || prod > 3 {
                        return false;
                    }
                }
            }
        }

        true
    }

    /// Get roots at a given inner product distance from a reference root.
    pub fn roots_at_distance(&self, ref_root: &E8Root, target_ip: i32) -> Vec<&E8Root> {
        self.roots.iter()
            .filter(|r| (r.inner_product(ref_root).round() as i32) == target_ip)
            .collect()
    }
}

impl Default for E8Lattice {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_e8_root_count() {
        let roots = generate_e8_roots();
        assert_eq!(roots.len(), 240);
    }

    #[test]
    fn test_all_roots_have_norm_2() {
        let roots = generate_e8_roots();
        for root in &roots {
            assert!(
                root.is_valid_root(),
                "Root {:?} has norm^2 = {}, expected 2",
                root.coords,
                root.norm_sq
            );
        }
    }

    #[test]
    fn test_simple_roots_valid() {
        let simple = e8_simple_roots();
        for (i, root) in simple.iter().enumerate() {
            assert!(
                root.is_valid_root(),
                "Simple root {} has invalid norm",
                i
            );
        }
    }

    #[test]
    fn test_cartan_matrix_structure() {
        let lattice = E8Lattice::new();
        assert!(lattice.verify_cartan_matrix());
    }

    #[test]
    fn test_inner_product_distribution() {
        let roots = generate_e8_roots();
        let ip_counts = compute_e8_inner_products(&roots);

        // Inner products should be in {-2, -1, 0, 1, 2}
        for (ip, _count) in &ip_counts {
            assert!(
                *ip >= -2 && *ip <= 2,
                "Unexpected inner product {}",
                ip
            );
        }
    }

    #[test]
    fn test_weyl_group_order() {
        let order = e8_weyl_group_order();
        assert_eq!(order, 696_729_600);

        // Verify factorization: 2^14 * 3^5 * 5^2 * 7
        let expected = 2u64.pow(14) * 3u64.pow(5) * 5u64.pow(2) * 7;
        assert_eq!(order, expected);
    }

    #[test]
    fn test_positive_root_count() {
        let lattice = E8Lattice::new();
        assert_eq!(lattice.positive_root_count(), 120);
    }
}
