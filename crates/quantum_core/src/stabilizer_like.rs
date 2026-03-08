//! Trait abstraction for stabilizer-like quantum error correction codes.
//!
//! Unifies BoxKiteStabilizer (Sedenion, 16D) and DekaVoudonStabilizer (1024D)
//! under a common interface. Both codes use non-associative algebraic structure
//! (zero-divisor parity checks) rather than standard Pauli stabilizers.
//!
//! See BIB-0313 (Kitaev 2003) for toric code foundations and
//! BIB-0314 (Pastawski et al. 2015) for holographic code structure.

use std::collections::HashSet;

/// A stabilizer-like code element with syndrome detection and distance metrics.
///
/// Unlike standard stabilizer codes where `[S_i, S_j] = 0` and syndromes
/// are binary, non-associative codes have continuous syndrome strengths
/// arising from alternativity violation density. The trait uses `f64`
/// syndrome values to accommodate both discrete (Sedenion box-kite)
/// and continuous (DekaVoudon holographic) detection models.
pub trait StabilizerLikeCode {
    /// Returns the set of qubit/axis indices covered by this code element.
    fn covered_indices(&self) -> &[usize];

    /// Computes syndrome strength given an error mask.
    ///
    /// Returns 0.0 for no detected error, 1.0 for maximal syndrome.
    /// Intermediate values represent partial syndrome activation
    /// (meaningful for holographic codes with graded parity).
    fn syndrome_strength(&self, error_mask: &HashSet<usize>) -> f64;

    /// Returns the code distance (minimum-weight uncorrectable error).
    ///
    /// For box-kite codes this is the K_{2,2,2} overlap threshold.
    /// For holographic codes this is the recursive chain depth.
    fn code_distance(&self) -> usize;

    /// Returns the number of parity-check constraints.
    fn num_checks(&self) -> usize;

    /// Returns the algebraic dimension of the parent algebra.
    fn algebra_dimension(&self) -> usize;
}

/// A composite code wrapping multiple stabilizer elements.
///
/// Models the full QEC code as a collection of stabilizer-like elements,
/// providing aggregate metrics (global syndrome, effective distance).
pub trait CompositeCode {
    /// Returns the number of stabilizer elements in the code.
    fn num_stabilizers(&self) -> usize;

    /// Global stability metric: 1.0 = no syndromes, 0.0 = fully triggered.
    fn global_stability(&self, error_mask: &HashSet<usize>) -> f64;

    /// Effective code distance (minimum across all stabilizers).
    fn effective_distance(&self) -> usize;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal test stabilizer for trait conformance.
    struct TestStabilizer {
        indices: Vec<usize>,
    }

    impl StabilizerLikeCode for TestStabilizer {
        fn covered_indices(&self) -> &[usize] {
            &self.indices
        }

        fn syndrome_strength(&self, error_mask: &HashSet<usize>) -> f64 {
            let hits = self
                .indices
                .iter()
                .filter(|i| error_mask.contains(i))
                .count();
            if self.indices.is_empty() {
                0.0
            } else {
                hits as f64 / self.indices.len() as f64
            }
        }

        fn code_distance(&self) -> usize {
            2
        }

        fn num_checks(&self) -> usize {
            self.indices.len().saturating_sub(1)
        }

        fn algebra_dimension(&self) -> usize {
            16
        }
    }

    #[test]
    fn test_stabilizer_trait_no_errors() {
        let stab = TestStabilizer {
            indices: vec![1, 2, 3, 4, 5, 6],
        };
        let empty = HashSet::new();
        assert_eq!(stab.syndrome_strength(&empty), 0.0);
        assert_eq!(stab.code_distance(), 2);
        assert_eq!(stab.num_checks(), 5);
        assert_eq!(stab.algebra_dimension(), 16);
    }

    #[test]
    fn test_stabilizer_trait_partial_error() {
        let stab = TestStabilizer {
            indices: vec![1, 2, 3, 4],
        };
        let mut errors = HashSet::new();
        errors.insert(2);
        errors.insert(4);
        let strength = stab.syndrome_strength(&errors);
        assert!((strength - 0.5).abs() < 1e-10, "Expected 0.5, got {}", strength);
    }

    #[test]
    fn test_stabilizer_trait_full_error() {
        let stab = TestStabilizer {
            indices: vec![1, 2, 3],
        };
        let errors: HashSet<usize> = [1, 2, 3].into_iter().collect();
        assert!((stab.syndrome_strength(&errors) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_covered_indices() {
        let stab = TestStabilizer {
            indices: vec![10, 20, 30],
        };
        assert_eq!(stab.covered_indices(), &[10, 20, 30]);
    }
}
