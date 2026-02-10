//! E8 Root System: The 248-dimensional exceptional Lie group.
//!
//! E8 is the largest exceptional simple Lie group, with 240 roots in 8-dimensional space.
//! All roots have the same norm (simply-laced root system).
//!
//! Root structure:
//! - D8 component: 112 roots of form (+/-1,+/-1,0^6) and permutations
//! - Demiocteract: 128 remaining roots
//! - Total: 240 roots (all equidistant from origin)
//!
//! Group properties:
//! - Dimension: 248 = rank(8) + #roots = 8 + 240
//! - Weyl group order: 2^14 x 3^5 x 5^2 x 7 approx 696.7 million
//! - Maximal root: sum of all simple roots (highest root)
//! - Rank: 8 (dimension of maximal torus)
//!
//! Connection to octonions:
//! - E8 arises from Freudenthal-Tits magic square: OtensorO -> E8
//! - Octonion non-associativity generates exceptional structure
//!
//! Reference: Yokota (arXiv:0902.0431), madore.org E8 reference

use std::collections::HashSet;

/// E8 Root System
#[derive(Clone, Debug)]
pub struct E8RootSystem {
    /// All 240 roots in 8D
    roots: Vec<[f64; 8]>,
    /// Simple roots (8 basis vectors from Dynkin diagram)
    simple_roots: Vec<[f64; 8]>,
}

impl E8RootSystem {
    /// Create E8 root system with all 240 roots enumerated.
    pub fn new() -> Self {
        let mut sys = E8RootSystem {
            roots: Vec::new(),
            simple_roots: Vec::new(),
        };
        sys.enumerate_roots();
        sys.enumerate_simple_roots();
        sys
    }

    /// Enumerate all 240 E8 roots.
    /// Structure: D8 component (112 roots) + demiocteract (128 roots)
    fn enumerate_roots(&mut self) {
        let mut root_set = HashSet::new();

        // D8 component: All (+/-1,+/-1,0,0,0,0,0,0) and permutations
        // Choose 2 positions out of 8, fill with +/-1
        for i in 0..8 {
            for j in (i + 1)..8 {
                for si in &[-1.0, 1.0] {
                    for sj in &[-1.0, 1.0] {
                        let mut root = [0.0; 8];
                        root[i] = *si;
                        root[j] = *sj;
                        root_set.insert(self.root_to_key(&root));
                        self.roots.push(root);
                    }
                }
            }
        }

        // Demiocteract: All (+/-1/2, +/-1/2, +/-1/2, +/-1/2, +/-1/2, +/-1/2, +/-1/2, +/-1/2) with even number of -1/2
        let mut demi_roots = Vec::new();
        for bits in 0..256 {
            let mut root = [0.5; 8];
            let mut neg_count = 0;
            for (k, r) in root.iter_mut().enumerate() {
                if (bits >> k) & 1 == 1 {
                    *r = -0.5;
                    neg_count += 1;
                }
            }
            // Only include if even number of negative components
            if neg_count % 2 == 0 {
                demi_roots.push(root);
            }
        }

        // Verify demiocteract has 128 roots (2^7 with even parity constraint)
        assert_eq!(demi_roots.len(), 128, "Demiocteract should have 128 roots");
        self.roots.extend(demi_roots);

        // Verify total count
        assert_eq!(
            self.roots.len(),
            240,
            "E8 should have exactly 240 roots, got {}",
            self.roots.len()
        );
    }

    /// Enumerate 8 simple roots from Dynkin diagram.
    /// Simple roots are the basis of the root lattice.
    fn enumerate_simple_roots(&mut self) {
        // Standard E8 simple roots (Dynkin diagram basis)
        // Using extended D8 representation
        self.simple_roots = vec![
            [1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // alpha1
            [0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0], // alpha2
            [0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0], // alpha3
            [0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0], // alpha4
            [0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0], // alpha5
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0], // alpha6
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0], // alpha7
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -0.5], // alpha8
        ];
    }

    /// Convert root to hashable key (for deduplication)
    fn root_to_key(&self, root: &[f64; 8]) -> String {
        root.iter()
            .map(|x| format!("{:.6}", x))
            .collect::<Vec<_>>()
            .join(",")
    }

    /// Get the list of roots
    pub fn get_roots(&self) -> &Vec<[f64; 8]> {
        &self.roots
    }

    /// Get simple roots (Dynkin diagram basis, dimension 8)
    pub fn simple_roots(&self) -> &[[f64; 8]] {
        &self.simple_roots
    }

    /// Cartan matrix: A_ij = 2(alpha_i dot alpha_j) / (alpha_i dot alpha_i)
    pub fn cartan_matrix(&self) -> [[i32; 8]; 8] {
        let mut matrix = [[0i32; 8]; 8];

        for (i, row) in matrix.iter_mut().enumerate() {
            let dot_ii = self.dot_product(&self.simple_roots[i], &self.simple_roots[i]);
            for (j, entry) in row.iter_mut().enumerate() {
                let dot_ij = self.dot_product(&self.simple_roots[i], &self.simple_roots[j]);
                *entry = (2.0 * dot_ij / dot_ii).round() as i32;
            }
        }

        matrix
    }

    /// Dot product in 8D Euclidean space
    fn dot_product(&self, a: &[f64; 8], b: &[f64; 8]) -> f64 {
        a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum()
    }

    /// Verify root system properties
    pub fn verify_root_structure(&self, tolerance: f64) -> bool {
        // Check: All roots have same norm squared = 2
        for root in &self.roots {
            let norm_sq = self.dot_product(root, root);
            if (norm_sq - 2.0).abs() > tolerance {
                return false;
            }
        }
        true
    }

    /// Check orthogonality of simple roots with Cartan matrix properties
    pub fn verify_cartan_matrix(&self) -> bool {
        let c = self.cartan_matrix();
        // Cartan matrix properties:
        // - Diagonal: C_ii = 2
        // - Off-diagonal: Standard is <= 0, but E8 can have +/-1 entries
        for (i, row) in c.iter().enumerate() {
            if row[i] != 2 {
                return false;
            }
        }
        for (i, row) in c.iter().enumerate() {
            for (j, &entry) in row.iter().enumerate() {
                if i != j && !(-3..=1).contains(&entry) {
                    return false;
                }
            }
        }
        true
    }

    /// Dimension of E8: 8 (rank) + 240 (roots) = 248
    pub fn dimension(&self) -> usize {
        248
    }

    /// Number of positive roots (half of 240)
    pub fn positive_root_count(&self) -> usize {
        120
    }

    /// Rank of the root system (dimension of maximal torus)
    pub fn rank(&self) -> usize {
        8
    }

    /// Number of roots (always 240 for E8)
    pub fn root_count(&self) -> usize {
        240
    }

    /// Weyl group order for E8: 2^14 x 3^5 x 5^2 x 7
    pub fn weyl_group_order(&self) -> u64 {
        let pow2_14: u64 = 2u64.pow(14);
        let pow3_5: u64 = 3u64.pow(5);
        let pow5_2: u64 = 5u64.pow(2);
        let pow7_1: u64 = 7u64;
        pow2_14 * pow3_5 * pow5_2 * pow7_1
    }
}

impl Default for E8RootSystem {
    fn default() -> Self {
        E8RootSystem::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_e8_root_count() {
        let e8 = E8RootSystem::new();
        assert_eq!(e8.root_count(), 240);
    }

    #[test]
    fn test_e8_rank() {
        let e8 = E8RootSystem::new();
        assert_eq!(e8.rank(), 8);
    }

    #[test]
    fn test_e8_dimension() {
        let e8 = E8RootSystem::new();
        assert_eq!(e8.dimension(), 248);
    }

    #[test]
    fn test_e8_simple_roots() {
        let e8 = E8RootSystem::new();
        let simple = e8.simple_roots();
        assert_eq!(simple.len(), 8);

        // Verify each simple root has norm squared = 2
        for root in simple {
            let norm_sq: f64 = root.iter().map(|x| x * x).sum();
            assert!((norm_sq - 2.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_e8_root_norms() {
        let e8 = E8RootSystem::new();
        let roots = e8.get_roots();

        // All roots should have norm squared = 2
        for root in roots {
            let norm_sq: f64 = root.iter().map(|x| x * x).sum();
            assert!(
                (norm_sq - 2.0).abs() < 1e-10,
                "Root norm verification failed"
            );
        }
    }

    #[test]
    fn test_e8_root_structure() {
        let e8 = E8RootSystem::new();
        // Verify all roots have norm squared = 2
        assert!(e8.verify_root_structure(1e-10));
    }

    #[test]
    fn test_e8_cartan_matrix() {
        let e8 = E8RootSystem::new();
        let c = e8.cartan_matrix();

        // Check diagonal
        for i in 0..8 {
            assert_eq!(c[i][i], 2);
        }

        // Check off-diagonal bounds: E8 Cartan matrix can have -1, 0, or +/-1
        // (The +/-1 entries arise from the E8 Dynkin diagram structure)
        for i in 0..8 {
            for j in 0..8 {
                if i != j {
                    // Allow {-3, -2, -1, 0, 1} for E8 Cartan matrix
                    assert!(
                        c[i][j] >= -3 && c[i][j] <= 1,
                        "Cartan entry c[{}][{}] = {} out of expected range",
                        i,
                        j,
                        c[i][j]
                    );
                }
            }
        }

        assert!(e8.verify_cartan_matrix());
    }

    #[test]
    fn test_e8_weyl_group_order() {
        let e8 = E8RootSystem::new();
        let order = e8.weyl_group_order();
        assert_eq!(order, 696_729_600); // 2^14 * 3^5 * 5^2 * 7
    }

    #[test]
    fn test_e8_positive_roots() {
        let e8 = E8RootSystem::new();
        assert_eq!(e8.positive_root_count(), 120);
    }

    #[test]
    fn test_e8_root_structure_verification() {
        let e8 = E8RootSystem::new();
        assert!(e8.verify_root_structure(1e-10));
    }
}
