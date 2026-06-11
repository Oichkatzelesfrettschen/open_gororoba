//! Hierarchical Codebook Logic for 256D -> 2048D Lattice Mappings.
//!
//! Implements the "predicate cut" filtration described in the analysis:
//! Lambda_256 <= Lambda_512 <= Lambda_1024 <= Lambda_2048 <= {-1, 0, 1}^8.
//!
//! # Hierarchy
//! - Base: Trinary vectors, even sum, even weight.
//! - 2048D: Base minus 139 forbidden prefixes.
//! - 1024D: 2048D intersected with {l_0 = -1} minus 70 prefixes.
//! - 512D: 1024D minus 6 forbidden regions (trie cuts).
//! - 256D: 512D minus 6 forbidden regions.
//!
//! # Typed Carriers (Layer 0)
//! A `TypedCarrier` X_n = (b, l) pairs a Cayley-Dickson basis element b with
//! its lattice vector l in the encoding dictionary. A `CarrierSet` collects
//! all carriers for a given dimension, providing O(1) lookup by basis index
//! and filtration membership queries.
//!
//! # Scalar Shadow
//! Implements the affine/linear action of the scalar shadow pi(b) on the lattice.

use std::collections::HashMap;

// Lambda filtration predicates (LatticeVector type + is_in_* family +
// verify_octonion_parity_constraints + enumerate_lambda_4096) live in
// the `lambda_predicates` submodule. Re-exports preserve the public API
// at algebra_analysis::codebook::{LatticeVector, is_in_*}.
pub mod lambda_predicates;
pub use lambda_predicates::{
    LatticeVector, enumerate_lambda_4096, is_in_base_universe, is_in_lambda_256, is_in_lambda_512,
    is_in_lambda_512_minus_k, is_in_lambda_1024, is_in_lambda_1024_minus_k, is_in_lambda_2048,
    is_in_lambda_2048_minus_k, is_in_lambda_4096, is_in_sbase_minus_k,
    verify_octonion_parity_constraints,
};

// Forbidden-prefix enumeration (ForbiddenFamily, ForbiddenPoint,
// classify_forbidden, enumerate_forbidden_2048) plus the generic
// enumerate_lattice_by_predicate helper and enumerate_lambda_256 live
// in the `forbidden_prefixes` submodule.
pub mod forbidden_prefixes;
pub use forbidden_prefixes::{
    ForbiddenFamily, ForbiddenPoint, classify_forbidden, enumerate_forbidden_2048,
    enumerate_lambda_256, enumerate_lattice_by_predicate,
};

// SliceCharacterization, characterize_pinned_slice (pinned-corner slice
// of Lambda_256), F_3 lattice arithmetic (lattice_add_f3,
// lattice_negate_f3, lattice_diff), and the scalar-shadow action
// (apply_scalar_shadow) live in the `lattice_arith` submodule.
pub mod lattice_arith;
pub use lattice_arith::{
    SliceCharacterization, apply_scalar_shadow, characterize_pinned_slice, lattice_add_f3,
    lattice_diff, lattice_negate_f3,
};

// Layer 0: TypedCarrier, CarrierSet, CarrierSetValidation live in
// the `carriers` submodule. Re-exported pub so external paths
// algebra_analysis::codebook::{TypedCarrier, CarrierSet,
// CarrierSetValidation} stay stable.
mod carriers;
pub use carriers::{CarrierSet, CarrierSetValidation, FiltrationTier, TypedCarrier};

// ============================================================================
// Layer 1: Encoding Dictionary Phi_n
// ============================================================================

/// The encoding dictionary Phi_n: {e_0, ..., e_{n-1}} -> Lambda_n.
///
/// This is a validated bijection between CD basis elements and lattice vectors.
/// It provides both forward (encode: basis -> lattice) and inverse
/// (decode: lattice -> basis) operations in O(1).
///
/// Construction requires that the underlying CarrierSet pass validation
/// (complete + injective). If validation fails, `try_from_carrier_set`
/// returns the validation errors.
#[derive(Debug, Clone)]
pub struct EncodingDictionary {
    /// The underlying carrier set (validated).
    carriers: CarrierSet,
    /// Inverse map: lattice vector -> basis index (O(1) decode).
    inverse: HashMap<LatticeVector, usize>,
}

impl EncodingDictionary {
    /// Attempt to build an encoding dictionary from a carrier set.
    /// Fails if the carrier set is not a valid bijection.
    pub fn try_from_carrier_set(cs: CarrierSet) -> Result<Self, CarrierSetValidation> {
        let validation = cs.validate();
        if !validation.is_valid_dictionary() {
            return Err(validation);
        }

        let inverse: HashMap<LatticeVector, usize> =
            cs.iter().map(|c| (c.lattice_vec, c.basis_index)).collect();

        Ok(Self {
            carriers: cs,
            inverse,
        })
    }

    /// Build from a basis_index -> `Vec<i32>` map (bridge from cd_external).
    /// Fails if the resulting carrier set is not a valid bijection.
    pub fn try_from_i32_map(
        dim: usize,
        map: &HashMap<usize, Vec<i32>>,
    ) -> Result<Self, CarrierSetValidation> {
        let cs = CarrierSet::from_i32_map(dim, map);
        Self::try_from_carrier_set(cs)
    }

    /// Build from pre-validated (basis_index, lattice_vector) pairs.
    pub fn try_from_pairs(
        dim: usize,
        pairs: &[(usize, LatticeVector)],
    ) -> Result<Self, CarrierSetValidation> {
        let cs = CarrierSet::from_lattice_vecs(dim, pairs);
        Self::try_from_carrier_set(cs)
    }

    /// The CD algebra dimension this dictionary encodes.
    pub fn dim(&self) -> usize {
        self.carriers.dim
    }

    /// Number of entries (should equal dim for a valid dictionary).
    pub fn len(&self) -> usize {
        self.carriers.len()
    }

    /// Whether the dictionary is empty.
    pub fn is_empty(&self) -> bool {
        self.carriers.is_empty()
    }

    /// Encode: Phi_n(basis_index) -> LatticeVector.
    /// Returns None if basis_index is not in [0, dim).
    pub fn encode(&self, basis_index: usize) -> Option<&LatticeVector> {
        self.carriers.get(basis_index).map(|c| &c.lattice_vec)
    }

    /// Decode: Phi_n^{-1}(lattice_vec) -> basis_index.
    /// Returns None if the lattice vector is not in the codebook.
    pub fn decode(&self, lattice_vec: &LatticeVector) -> Option<usize> {
        self.inverse.get(lattice_vec).copied()
    }

    /// Access the underlying carrier set.
    pub fn carrier_set(&self) -> &CarrierSet {
        &self.carriers
    }

    /// Iterate over all (basis_index, lattice_vector) pairs in order.
    pub fn iter(&self) -> impl Iterator<Item = (usize, &LatticeVector)> {
        self.carriers
            .iter()
            .map(|c| (c.basis_index, &c.lattice_vec))
    }

    /// Restrict this dictionary to carriers whose lattice vectors are in
    /// Lambda_{target_dim}. Returns a new (smaller) dictionary for the
    /// sub-codebook at the target filtration tier.
    ///
    /// Note: the returned dictionary has dim = target_dim, and its basis
    /// indices are the ORIGINAL indices from the parent dictionary. It will
    /// not pass completeness validation (missing basis indices are expected).
    pub fn restrict_to_lambda(&self, target_dim: usize) -> Vec<(usize, LatticeVector)> {
        self.carriers
            .iter()
            .filter(|c| c.is_in_lambda(target_dim))
            .map(|c| (c.basis_index, c.lattice_vec))
            .collect()
    }

    /// Compute the scalar shadow pi(b) for a basis element.
    /// Defined as sign(sum(lattice_vec)).
    pub fn scalar_shadow(&self, basis_index: usize) -> Option<i8> {
        self.encode(basis_index).map(|lv| {
            let s: i32 = lv.iter().map(|&x| x as i32).sum();
            if s > 0 {
                1
            } else if s < 0 {
                -1
            } else {
                0
            }
        })
    }
}

// Layer 2: Elevated Addition (lattice_add, try_narrow_to_lattice,
// ElevatedResult enum, ElevatedAdditionStats, ElevatedResultF3 enum,
// ElevatedAdditionStatsF3, plus four EncodingDictionary methods via
// a second impl block) live in the `elevated_addition` submodule.
// Public surface re-exported via pub use so
// algebra_analysis::codebook::{lattice_add, ElevatedResult, ...}
// paths remain stable for external callers.
mod elevated_addition;
pub use elevated_addition::{
    ElevatedAdditionStats, ElevatedAdditionStatsF3, ElevatedResult, ElevatedResultF3, lattice_add,
    try_narrow_to_lattice,
};

// Pure linear-algebra helpers (kahan_dot, kahan_norm_sq,
// gram_schmidt_basis, project_to_basis, find_pivot_columns_reduced,
// build_square_matrix, invert_nxn, mat_mul_nxn, det_nxn) live in the
// `linear_algebra` submodule. All callers live in the `coupling`
// submodule now; no parent-level use re-export needed.
mod linear_algebra;

// Multiplication-coupling analysis (BasisCouplingResult,
// MultiplicationCoupling, compute_multiplication_coupling) lives in
// the `coupling` submodule. Re-exported pub at parent scope so
// external callers can keep using
// `algebra_analysis::codebook::{BasisCouplingResult, ...}`.
mod coupling;
pub use coupling::{BasisCouplingResult, MultiplicationCoupling, compute_multiplication_coupling};

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::{
        linear_algebra::{det_nxn, gram_schmidt_basis, invert_nxn},
        *,
    };

    #[test]
    fn test_typed_carrier_from_i32_vec() {
        let c = TypedCarrier::from_i32_vec(0, &[-1, -1, -1, -1, 0, 0, 0, 0]);
        assert!(c.is_some());
        let c = c.unwrap();
        assert_eq!(c.basis_index, 0);
        assert_eq!(c.lattice_vec, [-1, -1, -1, -1, 0, 0, 0, 0]);
    }

    #[test]
    fn test_typed_carrier_rejects_out_of_range() {
        assert!(TypedCarrier::from_i32_vec(0, &[2, 0, 0, 0, 0, 0, 0, 0]).is_none());
        assert!(TypedCarrier::from_i32_vec(0, &[0, 0, -2, 0, 0, 0, 0, 0]).is_none());
    }

    #[test]
    fn test_typed_carrier_rejects_wrong_length() {
        assert!(TypedCarrier::from_i32_vec(0, &[0, 0, 0]).is_none());
        assert!(TypedCarrier::from_i32_vec(0, &[0; 9]).is_none());
    }

    #[test]
    fn test_carrier_filtration_tier() {
        // This vector should be in Lambda_256: l_0=-1, l_1=-1, ...
        let c = TypedCarrier::new(0, [-1, -1, -1, -1, 0, 0, 0, 0]);
        let tier = c.filtration_tier();
        assert_eq!(tier, FiltrationTier::Lambda256);
    }

    #[test]
    fn test_carrier_is_in_lambda() {
        let c = TypedCarrier::new(0, [-1, -1, -1, -1, 0, 0, 0, 0]);
        // Lambda_256 is the most restrictive; membership implies all larger sets.
        assert!(c.is_in_lambda(256));
        assert!(c.is_in_lambda(512));
        assert!(c.is_in_lambda(1024));
        assert!(c.is_in_lambda(2048));
    }

    #[test]
    fn test_carrier_set_from_lattice_vecs() {
        let pairs = vec![
            (0, [-1, -1, -1, -1, 0, 0, 0, 0]),
            (1, [-1, -1, 0, 0, -1, -1, 0, 0]),
            (2, [-1, -1, 0, 0, 0, 0, -1, -1]),
        ];
        let cs = CarrierSet::from_lattice_vecs(3, &pairs);
        assert_eq!(cs.len(), 3);
        assert!(!cs.is_empty());
        assert!(cs.get(0).is_some());
        assert!(cs.get(1).is_some());
        assert!(cs.get(2).is_some());
        assert!(cs.get(3).is_none());
    }

    #[test]
    fn test_carrier_set_validation_complete() {
        let pairs = vec![
            (0, [-1, -1, -1, -1, 0, 0, 0, 0]),
            (1, [-1, -1, 0, 0, -1, -1, 0, 0]),
        ];
        let cs = CarrierSet::from_lattice_vecs(2, &pairs);
        let v = cs.validate();
        assert!(v.is_complete);
        assert!(v.is_injective);
        assert!(v.is_valid_dictionary());
    }

    #[test]
    fn test_carrier_set_validation_missing() {
        let pairs = vec![
            (0, [-1, -1, -1, -1, 0, 0, 0, 0]),
            // basis_index 1 is missing
            (2, [-1, -1, 0, 0, -1, -1, 0, 0]),
        ];
        let cs = CarrierSet::from_lattice_vecs(3, &pairs);
        let v = cs.validate();
        assert!(!v.is_complete);
        assert_eq!(v.missing_basis_indices, vec![1]);
        assert!(!v.is_valid_dictionary());
    }

    #[test]
    fn test_carrier_set_validation_duplicate_lattice() {
        let same_vec = [-1, -1, -1, -1, 0, 0, 0, 0];
        let pairs = vec![
            (0, same_vec),
            (1, same_vec), // duplicate lattice vector
        ];
        let cs = CarrierSet::from_lattice_vecs(2, &pairs);
        let v = cs.validate();
        assert!(v.is_complete);
        assert!(!v.is_injective);
        assert_eq!(v.duplicate_lattice_pairs.len(), 1);
        assert!(!v.is_valid_dictionary());
    }

    #[test]
    fn test_carrier_set_filter_to_lambda() {
        // Mix: one vector in Lambda_256, one not (l_0 = 0, fails Lambda_1024).
        let pairs = vec![
            (0, [-1, -1, -1, -1, 0, 0, 0, 0]), // in Lambda_256
            (1, [0, -1, 0, -1, 0, -1, 0, -1]), // base only (l_0 = 0)
        ];
        let cs = CarrierSet::from_lattice_vecs(2, &pairs);
        let in_256 = cs.filter_to_lambda(256);
        assert_eq!(in_256.len(), 1);
        assert_eq!(in_256[0].basis_index, 0);
    }

    #[test]
    fn test_carrier_set_tier_histogram() {
        let pairs = vec![
            (0, [-1, -1, -1, -1, 0, 0, 0, 0]),   // Lambda_256
            (1, [-1, -1, -1, -1, -1, -1, 0, 0]), // Lambda_256
        ];
        let cs = CarrierSet::from_lattice_vecs(2, &pairs);
        let hist = cs.tier_histogram();
        assert_eq!(hist.get(&FiltrationTier::Lambda256), Some(&2));
    }

    #[test]
    fn test_carrier_set_from_i32_map() {
        let mut map = HashMap::new();
        map.insert(0, vec![-1, -1, -1, -1, 0, 0, 0, 0]);
        map.insert(1, vec![-1, -1, 0, 0, -1, -1, 0, 0]);
        let cs = CarrierSet::from_i32_map(2, &map);
        assert_eq!(cs.len(), 2);
        let v = cs.validate();
        assert!(v.is_valid_dictionary());
    }

    #[test]
    fn test_filtration_nesting() {
        // Any vector in Lambda_256 must also be in Lambda_512, 1024, 2048, Base.
        let v: LatticeVector = [-1, -1, -1, -1, 0, 0, 0, 0];
        if is_in_lambda_256(&v) {
            assert!(is_in_lambda_512(&v));
            assert!(is_in_lambda_1024(&v));
            assert!(is_in_lambda_2048(&v));
            assert!(is_in_base_universe(&v));
        }
    }

    #[test]
    fn test_base_universe_parity() {
        // Even sum + even weight + trinary + l_0 != 1
        assert!(is_in_base_universe(&[-1, -1, 0, 0, 0, 0, 0, 0])); // sum=-2, wt=2
        assert!(is_in_base_universe(&[0, 0, 0, 0, 0, 0, 0, 0])); // sum=0, wt=0
        assert!(!is_in_base_universe(&[1, 0, 0, 0, 0, 0, 0, 0])); // l_0=1 forbidden
        assert!(!is_in_base_universe(&[-1, 0, 0, 0, 0, 0, 0, 0])); // sum=-1 odd
    }

    #[test]
    fn test_scalar_shadow_add() {
        let v: LatticeVector = [-1, 0, 1, 0, -1, 0, 1, 0];
        let shifted = apply_scalar_shadow(&v, 1, "add");
        assert_eq!(shifted, [0, 1, 2, 1, 0, 1, 2, 1]);
    }

    #[test]
    fn test_scalar_shadow_mul() {
        let v: LatticeVector = [-1, 0, 1, 0, -1, 0, 1, 0];
        let scaled = apply_scalar_shadow(&v, -1, "mul");
        assert_eq!(scaled, [1, 0, -1, 0, 1, 0, -1, 0]);
    }

    // ---- EncodingDictionary tests ----

    fn sample_dictionary_4() -> EncodingDictionary {
        let pairs = vec![
            (0, [-1, -1, -1, -1, 0, 0, 0, 0]),
            (1, [-1, -1, 0, 0, -1, -1, 0, 0]),
            (2, [-1, -1, 0, 0, 0, 0, -1, -1]),
            (3, [-1, 0, -1, 0, -1, 0, -1, 0]),
        ];
        EncodingDictionary::try_from_pairs(4, &pairs).unwrap()
    }

    #[test]
    fn test_encoding_dictionary_encode_decode() {
        let dict = sample_dictionary_4();
        assert_eq!(dict.dim(), 4);
        assert_eq!(dict.len(), 4);

        // Forward: encode
        let lv = dict.encode(0).unwrap();
        assert_eq!(*lv, [-1, -1, -1, -1, 0, 0, 0, 0]);

        // Inverse: decode
        let idx = dict.decode(&[-1, -1, 0, 0, -1, -1, 0, 0]).unwrap();
        assert_eq!(idx, 1);
    }

    #[test]
    fn test_encoding_dictionary_round_trip() {
        let dict = sample_dictionary_4();
        for b in 0..4 {
            let lv = dict.encode(b).unwrap();
            let decoded = dict.decode(lv).unwrap();
            assert_eq!(decoded, b, "round-trip failed for basis {b}");
        }
    }

    #[test]
    fn test_encoding_dictionary_decode_missing() {
        let dict = sample_dictionary_4();
        let missing = [0, 0, 0, 0, 0, 0, 0, 0];
        assert!(dict.decode(&missing).is_none());
    }

    #[test]
    fn test_encoding_dictionary_rejects_incomplete() {
        let pairs = vec![
            (0, [-1, -1, -1, -1, 0, 0, 0, 0]),
            // basis 1 missing
            (2, [-1, -1, 0, 0, 0, 0, -1, -1]),
        ];
        let result = EncodingDictionary::try_from_pairs(3, &pairs);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(!err.is_complete);
        assert_eq!(err.missing_basis_indices, vec![1]);
    }

    #[test]
    fn test_encoding_dictionary_rejects_non_injective() {
        let same_vec = [-1, -1, -1, -1, 0, 0, 0, 0];
        let pairs = vec![(0, same_vec), (1, same_vec)];
        let result = EncodingDictionary::try_from_pairs(2, &pairs);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(!err.is_injective);
    }

    #[test]
    fn test_encoding_dictionary_scalar_shadow() {
        let dict = sample_dictionary_4();
        // Basis 0: [-1,-1,-1,-1,0,0,0,0] -> sum=-4, signum=-1
        assert_eq!(dict.scalar_shadow(0), Some(-1));
        // Basis 3: [-1,0,-1,0,-1,0,-1,0] -> sum=-4, signum=-1
        assert_eq!(dict.scalar_shadow(3), Some(-1));
    }

    #[test]
    fn test_encoding_dictionary_restrict_to_lambda() {
        let dict = sample_dictionary_4();
        let restricted = dict.restrict_to_lambda(256);
        // All our test vectors have l_0=-1, l_1=-1 which should be in Lambda_256.
        // Let's verify at least some pass.
        assert!(!restricted.is_empty());
    }

    #[test]
    fn test_encoding_dictionary_from_i32_map() {
        let mut map = HashMap::new();
        map.insert(0, vec![-1, -1, -1, -1, 0, 0, 0, 0]);
        map.insert(1, vec![-1, -1, 0, 0, -1, -1, 0, 0]);
        let dict = EncodingDictionary::try_from_i32_map(2, &map).unwrap();
        assert_eq!(dict.len(), 2);
        assert_eq!(dict.encode(0).unwrap(), &[-1, -1, -1, -1, 0, 0, 0, 0]);
    }

    #[test]
    fn test_encoding_dictionary_iter() {
        let dict = sample_dictionary_4();
        let entries: Vec<_> = dict.iter().collect();
        assert_eq!(entries.len(), 4);
        // Should be sorted by basis_index
        assert_eq!(entries[0].0, 0);
        assert_eq!(entries[1].0, 1);
        assert_eq!(entries[2].0, 2);
        assert_eq!(entries[3].0, 3);
    }

    // ================================================================
    // 2048D Forbidden Prefix Enumeration Tests
    // ================================================================

    #[test]
    fn test_forbidden_2048_count() {
        let forbidden = enumerate_forbidden_2048();
        eprintln!("Forbidden 2048D points: {}", forbidden.len());
        assert_eq!(
            forbidden.len(),
            139,
            "Base universe minus Lambda_2048 should have exactly 139 points"
        );
    }

    #[test]
    fn test_forbidden_2048_all_in_base_universe() {
        let forbidden = enumerate_forbidden_2048();
        for fp in &forbidden {
            assert!(
                is_in_base_universe(&fp.vector),
                "Forbidden point {:?} should be in base universe",
                fp.vector
            );
        }
    }

    #[test]
    fn test_forbidden_2048_none_in_lambda() {
        let forbidden = enumerate_forbidden_2048();
        for fp in &forbidden {
            assert!(
                !is_in_lambda_2048(&fp.vector),
                "Forbidden point {:?} should NOT be in Lambda_2048",
                fp.vector
            );
        }
    }

    #[test]
    fn test_forbidden_2048_family_counts() {
        let forbidden = enumerate_forbidden_2048();
        let n_p1 = forbidden
            .iter()
            .filter(|f| f.family == ForbiddenFamily::Prefix011)
            .count();
        let n_p2 = forbidden
            .iter()
            .filter(|f| f.family == ForbiddenFamily::Prefix01011)
            .count();
        let n_p3 = forbidden
            .iter()
            .filter(|f| f.family == ForbiddenFamily::Prefix010101)
            .count();

        eprintln!(
            "Forbidden families: Prefix011={}, Prefix01011={}, Prefix010101={}",
            n_p1, n_p2, n_p3
        );
        eprintln!(
            "  Total: {} (= {} + {} + {})",
            n_p1 + n_p2 + n_p3,
            n_p1,
            n_p2,
            n_p3
        );

        // All families should be non-empty
        assert!(n_p1 > 0, "Prefix011 family should be non-empty");
        assert!(n_p2 > 0, "Prefix01011 family should be non-empty");
        assert!(n_p3 > 0, "Prefix010101 family should be non-empty");

        // Families should partition the forbidden set (mutually exclusive)
        assert_eq!(
            n_p1 + n_p2 + n_p3,
            139,
            "Three families should partition all 139 forbidden points"
        );
    }

    #[test]
    fn test_forbidden_2048_families_mutually_exclusive() {
        let forbidden = enumerate_forbidden_2048();
        // Verify mutual exclusivity by construction:
        // Pattern 1 has l_2=1, patterns 2 & 3 have l_2=0
        // Pattern 2 has l_4=1, pattern 3 has l_4=0
        for fp in &forbidden {
            let v = &fp.vector;
            match fp.family {
                ForbiddenFamily::Prefix011 => {
                    assert_eq!(v[0], 0);
                    assert_eq!(v[1], 1);
                    assert_eq!(v[2], 1);
                }
                ForbiddenFamily::Prefix01011 => {
                    assert_eq!(v[0], 0);
                    assert_eq!(v[1], 1);
                    assert_eq!(v[2], 0);
                    assert_eq!(v[3], 1);
                    assert_eq!(v[4], 1);
                }
                ForbiddenFamily::Prefix010101 => {
                    assert_eq!(v[0], 0);
                    assert_eq!(v[1], 1);
                    assert_eq!(v[2], 0);
                    assert_eq!(v[3], 1);
                    assert_eq!(v[4], 0);
                    assert_eq!(v[5], 1);
                }
            }
        }
    }

    #[test]
    fn test_forbidden_2048_exhaustive_coverage() {
        // Verify that enumerate_forbidden_2048 and is_in_lambda_2048 are consistent:
        // every base universe point is either in Lambda_2048 OR in the forbidden set.
        let forbidden = enumerate_forbidden_2048();
        let forbidden_set: std::collections::HashSet<LatticeVector> =
            forbidden.iter().map(|f| f.vector).collect();

        let mut n_base = 0usize;
        let mut n_lambda = 0usize;
        let mut n_forbidden = 0usize;

        for code in 0..3u32.pow(8) {
            let mut v = [0i8; 8];
            let mut c = code;
            for coord in &mut v {
                *coord = (c % 3) as i8 - 1;
                c /= 3;
            }
            if is_in_base_universe(&v) {
                n_base += 1;
                if is_in_lambda_2048(&v) {
                    n_lambda += 1;
                    assert!(
                        !forbidden_set.contains(&v),
                        "Lambda_2048 point should not be in forbidden set"
                    );
                } else {
                    n_forbidden += 1;
                    assert!(
                        forbidden_set.contains(&v),
                        "Non-Lambda_2048 base point should be in forbidden set"
                    );
                }
            }
        }

        eprintln!(
            "Exhaustive scan: base={}, lambda_2048={}, forbidden={}",
            n_base, n_lambda, n_forbidden
        );
        assert_eq!(n_forbidden, 139);
        assert_eq!(n_base, n_lambda + n_forbidden);
    }

    // ================================================================
    // Lambda enumeration from predicates
    // ================================================================

    #[test]
    fn test_enumerate_lambda_256_count() {
        // Enumerate Lambda_256 from predicates alone (no CSV data).
        // The predicate chain is an approximation of the true trie filtration;
        // the CSV-based ground truth has exactly 256 points. This test documents
        // how many the predicate gives.
        let points = enumerate_lambda_256();
        eprintln!("Lambda_256 from predicates: {} points", points.len());

        // The predicates may not give exactly 256 due to omitted singleton
        // exceptions (see lambda_1024 line 98 comment). Document the count.
        // If the predicate is exact, this will be 256.
        assert!(
            points.len() >= 256,
            "Predicates should accept at least 256 points (supersets are expected)"
        );
        // Upper bound: no more than Lambda_512's count (which should be <= 512).
        let p512 = enumerate_lattice_by_predicate(is_in_lambda_512);
        eprintln!("Lambda_512 from predicates: {} points", p512.len());
        assert!(
            points.len() <= p512.len(),
            "Lambda_256 must be a subset of Lambda_512"
        );
    }

    #[test]
    fn test_enumerate_filtration_nesting() {
        // Verify strict nesting Lambda_256 <= Lambda_512 <= Lambda_1024 <= Lambda_2048 <= Base
        let base = enumerate_lattice_by_predicate(is_in_base_universe);
        let l2048 = enumerate_lattice_by_predicate(is_in_lambda_2048);
        let l1024 = enumerate_lattice_by_predicate(is_in_lambda_1024);
        let l512 = enumerate_lattice_by_predicate(is_in_lambda_512);
        let l256 = enumerate_lambda_256();

        eprintln!(
            "Filtration counts: base={}, 2048={}, 1024={}, 512={}, 256={}",
            base.len(),
            l2048.len(),
            l1024.len(),
            l512.len(),
            l256.len()
        );

        // Strict nesting
        assert!(l256.len() < l512.len(), "Lambda_256 < Lambda_512");
        assert!(l512.len() < l1024.len(), "Lambda_512 < Lambda_1024");
        assert!(l1024.len() < l2048.len(), "Lambda_1024 < Lambda_2048");
        assert!(l2048.len() < base.len(), "Lambda_2048 < Base");

        // Subset inclusion
        let l2048_set: std::collections::HashSet<LatticeVector> = l2048.iter().copied().collect();
        for v in &l1024 {
            assert!(
                l2048_set.contains(v),
                "Lambda_1024 point not in Lambda_2048"
            );
        }
        let l512_set: std::collections::HashSet<LatticeVector> = l512.iter().copied().collect();
        for v in &l256 {
            assert!(l512_set.contains(v), "Lambda_256 point not in Lambda_512");
        }
    }

    // ================================================================
    // 32-point slice characterization (Task #115)
    // ================================================================

    #[test]
    fn test_pinned_slice_prefix_4_count() {
        // The "pinned corner" with prefix (-1,-1,-1,-1) and free tail in {-1,0,1}^4.
        // Since all trie-cut exclusions are vacuously satisfied for this prefix,
        // the slice = base_universe restricted to {l[0..4]=(-1,-1,-1,-1)}.
        //
        // The tail must have: even sum AND even nonzero count.
        // Weight 0: (0,0,0,0) -- 1 vector
        // Weight 2: C(4,2)*2^2 = 24 vectors (all even sum automatically)
        // Weight 4: 2^4 = 16 vectors (all even sum automatically)
        // Total: 41 points.
        let char = characterize_pinned_slice(4);
        eprintln!("Pinned-corner prefix=4: {} points", char.count);
        eprintln!("Tail weight histogram: {:?}", char.tail_weight_histogram);

        // The count should be 41 (not 32) from pure predicates.
        // The "32-point slice" in the literature is the lex-first 32 of
        // Lambda_256 from CSV data, which happens to be a subset of these 41.
        assert_eq!(
            char.count, 41,
            "Pinned prefix (-1,-1,-1,-1) slice should have 41 points"
        );

        // Weight distribution: w=0 -> 1, w=2 -> 24, w=4 -> 16
        assert_eq!(char.tail_weight_histogram, vec![(0, 1), (2, 24), (4, 16)]);
    }

    #[test]
    fn test_pinned_slice_prefix_4_geometry() {
        // Characterize the geometric structure of the 41-point slice.
        // All points share the common prefix (-1,-1,-1,-1), so pairwise
        // distances only depend on the tail coordinates (l_4..l_7).
        let char = characterize_pinned_slice(4);

        eprintln!(
            "Distance histogram (d^2, count): {:?}",
            char.distance_histogram
        );
        eprintln!(
            "Inner product histogram (ip, count): {:?}",
            char.inner_product_histogram
        );

        // All pairs: C(41,2) = 820
        let total_pairs: usize = char.distance_histogram.iter().map(|(_, c)| c).sum();
        assert_eq!(total_pairs, 41 * 40 / 2, "Should have C(41,2) pairs");

        // The prefix contributes 4 to the squared distance between any two
        // distinct points (since the prefix is identical). Wait -- no, the prefix
        // is IDENTICAL so it contributes 0 to squared distance. The distance
        // is entirely from the tail differences.
        //
        // Minimum nonzero d^2 = 2 (two tail coords differ by 1 each).
        // Maximum d^2 = 4*4 = 16 (all four tail coords differ by 2 each).
        //
        // But actually d^2 computed over the full 8D vector: since prefix is identical,
        // the first 4 coords contribute 0, and we only get distance from the tail.

        // Verify all distances are from tail-only differences
        for &(d2, _count) in &char.distance_histogram {
            assert!(d2 > 0, "No zero distances (points are distinct)");
            assert!(
                d2 <= 16,
                "Max d^2 = 4*2^2 = 16 (all tail coords differ by 2)"
            );
        }
    }

    #[test]
    fn test_pinned_slice_prefix_4_tail_structure() {
        // The 41 tail patterns in {-1,0,1}^4 with even sum and even weight
        // form a recognizable combinatorial object.
        //
        // The weight-4 subset (16 points in {-1,+1}^4 with even sum) is
        // exactly the D4 root system: the 8 vectors with an even number of
        // minus signs, plus the 8 with an odd number = all 16 of {-1,+1}^4.
        // Actually {-1,+1}^4 has all even sums (sum is always even for 4 terms
        // of +/-1), so all 16 qualify. This is the vertex set of a 4-cube (tesseract).
        //
        // The weight-2 subset (24 points with exactly 2 nonzeros in {-1,+1})
        // corresponds to the edge midpoints of the tesseract.
        //
        // The weight-0 subset is just the origin.
        let char = characterize_pinned_slice(4);

        // Extract the weight-4 tail patterns
        let w4: Vec<&LatticeVector> = char
            .tail_patterns
            .iter()
            .filter(|t| t[4..].iter().filter(|&&x| x != 0).count() == 4)
            .collect();
        assert_eq!(
            w4.len(),
            16,
            "16 weight-4 tail patterns (full tesseract vertices)"
        );

        // Verify these are exactly {-1,+1}^4 (with prefix positions zeroed)
        for t in &w4 {
            for i in 4..8 {
                assert!(
                    t[i] == -1 || t[i] == 1,
                    "Weight-4 tail should be +/-1 in free coordinates"
                );
            }
        }

        // Extract weight-2 patterns: C(4,2) * 4 = 24
        let w2: Vec<&LatticeVector> = char
            .tail_patterns
            .iter()
            .filter(|t| t[4..].iter().filter(|&&x| x != 0).count() == 2)
            .collect();
        assert_eq!(w2.len(), 24, "24 weight-2 tail patterns");

        // Each weight-2 pattern has exactly 2 nonzero coords in positions 4-7
        for t in &w2 {
            let nz_positions: Vec<usize> = (4..8).filter(|&i| t[i] != 0).collect();
            assert_eq!(nz_positions.len(), 2);
        }
    }

    #[test]
    fn test_pinned_slice_prefix_4_inner_products() {
        // Analyze inner products to detect polytope structure.
        // For the full 8D vectors, <v, w> = (-1)^2 * 4 + <tail_v, tail_w>
        //                                = 4 + <tail_v, tail_w>
        // So the inner product structure is shifted by +4 from the tail-only products.
        let char = characterize_pinned_slice(4);

        // Verify the shift: all inner products should be >= 4 - 4 = 0
        // (tail inner product minimum is -4 when all signs flip).
        // Actually: min tail ip is -4 (w4 vs w4 with all signs flipped),
        // so min full ip = 4 + (-4) = 0.
        for &(ip, _count) in &char.inner_product_histogram {
            assert!(
                ip >= 0,
                "Full inner product should be >= 0 (prefix contributes +4)"
            );
        }

        // The inner product with self is: 4 + sum(tail_i^2).
        // For weight-4: self-ip = 4 + 4 = 8
        // For weight-2: self-ip = 4 + 2 = 6
        // For weight-0: self-ip = 4 + 0 = 4
        // (Self inner products are not in the histogram since we skip i==j)

        // The maximum inter-point ip should be 8 (two weight-4 vectors that agree)
        // minus the common prefix = wait, two identical weight-4 vectors ARE the
        // same point. The max inter-point ip for distinct points:
        // weight-4 vs weight-4 with 3 signs matching, 1 flipped: ip = 4 + (3-1) = 6
        // Actually: if tail_v and tail_w are both in {-1,+1}^4, their ip = n_agree - n_disagree
        // where n_agree + n_disagree = 4.
        // ip = n_agree - (4 - n_agree) = 2*n_agree - 4.
        // Max (distinct): n_agree=3 -> ip=2, so full ip = 4+2 = 6.
        // Min: n_agree=0 -> ip=-4, so full ip = 4-4 = 0.
        // Cross (w4 vs w2): tail ip can be at most 2 (2 nonzeros agree).
        eprintln!(
            "Inner product histogram: {:?}",
            char.inner_product_histogram
        );
    }

    #[test]
    fn test_pinned_slice_trie_vacuity() {
        // Verify the key mathematical claim: for any vector with l[0..4] = (-1,-1,-1,-1),
        // the trie-cut exclusions at EVERY level are vacuously satisfied.
        // Therefore is_in_lambda_256(v) reduces to is_in_base_universe(v).
        let mut n_tested = 0usize;
        let mut n_agree = 0usize;

        for code in 0..3u32.pow(4) {
            let mut v = [-1i8, -1, -1, -1, 0, 0, 0, 0];
            let mut c = code;
            for vi in v[4..8].iter_mut() {
                *vi = (c % 3) as i8 - 1;
                c /= 3;
            }
            n_tested += 1;
            let in_base = is_in_base_universe(&v);
            let in_256 = is_in_lambda_256(&v);
            if in_base == in_256 {
                n_agree += 1;
            }
        }

        eprintln!("Trie vacuity check: {}/{} agree", n_agree, n_tested);
        assert_eq!(
            n_agree, n_tested,
            "For prefix (-1,-1,-1,-1), is_in_lambda_256 == is_in_base_universe everywhere"
        );
    }

    #[test]
    fn test_pinned_slice_weight4_is_tesseract() {
        // The 16 weight-4 tail vectors are the vertices of a 4-dimensional
        // hypercube (tesseract) centered at the origin, with vertices at {-1,+1}^4.
        //
        // The tesseract has these graph-theoretic properties:
        // - 16 vertices
        // - Each vertex has 4 nearest neighbors (Hamming distance 1 = flip one sign)
        // - 32 edges (16 * 4 / 2)
        // - The adjacency graph is the 4-cube graph Q_4
        let _char = characterize_pinned_slice(4);

        // Extract weight-4 points (full 8D vectors)
        let all_256 = enumerate_lambda_256();
        let w4_points: Vec<LatticeVector> = all_256
            .iter()
            .filter(|v| v[..4] == [-1, -1, -1, -1])
            .filter(|v| v[4..].iter().filter(|&&x| x != 0).count() == 4)
            .copied()
            .collect();

        assert_eq!(w4_points.len(), 16);

        // Compute adjacency: two vertices are adjacent if they differ in exactly
        // one tail coordinate (squared distance = 4 in that coordinate = 2^2).
        // Full 8D d^2 = 0 (prefix) + 4 (one flip) = 4.
        let mut edge_count = 0usize;
        let mut degree_counts = [0usize; 16];
        for i in 0..16 {
            for j in (i + 1)..16 {
                let d2: i32 = (0..8)
                    .map(|k| {
                        let d = w4_points[i][k] as i32 - w4_points[j][k] as i32;
                        d * d
                    })
                    .sum();
                if d2 == 4 {
                    edge_count += 1;
                    degree_counts[i] += 1;
                    degree_counts[j] += 1;
                }
            }
        }

        eprintln!(
            "Tesseract verification: {} vertices, {} edges",
            16, edge_count
        );
        assert_eq!(edge_count, 32, "Tesseract has 32 edges");
        for (idx, &deg) in degree_counts.iter().enumerate() {
            assert_eq!(deg, 4, "Vertex {} has degree {} (expected 4)", idx, deg);
        }
    }

    // ================================================================
    // Layer 2: Elevated addition tests
    // ================================================================

    /// Build a small synthetic dictionary for testing elevated addition.
    /// Uses 4 basis elements mapping to distinct Lambda_256 vectors.
    fn make_test_dictionary_4() -> EncodingDictionary {
        // Pick 4 vectors from Lambda_256 that are well-separated
        let pairs: Vec<(usize, LatticeVector)> = vec![
            (0, [-1, -1, -1, -1, 0, 0, 0, 0]),   // weight 4, sum -4
            (1, [-1, -1, -1, -1, -1, -1, 0, 0]), // weight 6, sum -6
            (2, [-1, -1, -1, -1, -1, 1, 0, 0]),  // weight 6, sum -4
            (3, [-1, -1, -1, -1, 0, 0, -1, -1]), // weight 6, sum -6
        ];
        EncodingDictionary::try_from_pairs(4, &pairs).unwrap()
    }

    #[test]
    fn test_lattice_add_basic() {
        let a: LatticeVector = [-1, -1, 0, 0, 0, 0, 0, 0];
        let b: LatticeVector = [0, 0, -1, -1, 0, 0, 0, 0];
        let sum = lattice_add(&a, &b);
        assert_eq!(sum, [-1, -1, -1, -1, 0, 0, 0, 0]);
    }

    #[test]
    fn test_lattice_add_overflow() {
        // Two -1s add to -2, which leaves {-1,0,1}^8
        let a: LatticeVector = [-1, -1, 0, 0, 0, 0, 0, 0];
        let b: LatticeVector = [-1, -1, 0, 0, 0, 0, 0, 0];
        let sum = lattice_add(&a, &b);
        assert_eq!(sum[0], -2, "Sum should be -2, outside trinary range");
        assert!(try_narrow_to_lattice(&sum).is_none());
    }

    #[test]
    fn test_try_narrow_to_lattice() {
        assert_eq!(
            try_narrow_to_lattice(&[-1, 0, 1, 0, 0, 0, 0, 0]),
            Some([-1, 0, 1, 0, 0, 0, 0, 0])
        );
        assert!(try_narrow_to_lattice(&[2, 0, 0, 0, 0, 0, 0, 0]).is_none());
        assert!(try_narrow_to_lattice(&[0, 0, 0, 0, 0, 0, 0, -2]).is_none());
    }

    #[test]
    fn test_elevated_add_in_codebook() {
        let dict = make_test_dictionary_4();
        // Phi(0) = (-1,-1,-1,-1,0,0,0,0) + Phi(0) = (-2,-2,-2,-2,0,0,0,0)
        // This overflows trinary -> OutOfBounds
        let r = dict.elevated_add(0, 0).unwrap();
        assert!(
            matches!(r, ElevatedResult::OutOfBounds { .. }),
            "Self-addition of (-1,-1,-1,-1,...) overflows"
        );
    }

    #[test]
    fn test_elevated_add_commutativity() {
        // lattice_add is inherently commutative (integer addition)
        let dict = make_test_dictionary_4();
        for a in 0..4 {
            for b in 0..4 {
                let r_ab = dict.elevated_add(a, b).unwrap();
                let r_ba = dict.elevated_add(b, a).unwrap();
                assert_eq!(
                    r_ab, r_ba,
                    "Elevated addition should be commutative: ({}, {})",
                    a, b
                );
            }
        }
    }

    #[test]
    fn test_elevated_addition_stats_synthetic() {
        let dict = make_test_dictionary_4();
        let stats = dict.elevated_addition_stats();

        eprintln!("Synthetic 4-element dictionary stats:");
        eprintln!(
            "  total_pairs={}, in_codebook={}, out_of_codebook={}, out_of_bounds={}",
            stats.total_pairs, stats.in_codebook, stats.out_of_codebook, stats.out_of_bounds
        );
        eprintln!(
            "  closure_rate={:.3}, commutative={}, identities={}",
            stats.closure_rate, stats.is_commutative, stats.identity_count
        );

        assert_eq!(stats.total_pairs, 16, "4x4 = 16 pairs");
        assert!(stats.is_commutative, "Lattice addition is commutative");
        assert_eq!(
            stats.in_codebook + stats.out_of_codebook + stats.out_of_bounds,
            16
        );
    }

    #[test]
    fn test_translation_orbit() {
        let dict = make_test_dictionary_4();
        for b in 0..4 {
            let orbit = dict.translation_orbit(b);
            eprintln!(
                "Translation orbit of b={}: {} pairs in codebook",
                b,
                orbit.len()
            );
            // Each orbit entry (a, c) means Phi(a) + Phi(b) = Phi(c)
            for &(a, c) in &orbit {
                assert!(c < 4, "Decoded index should be valid");
                let result = dict.elevated_add(a, b).unwrap();
                assert!(
                    matches!(result, ElevatedResult::InCodebook { decoded_index, .. }
                    if decoded_index == c)
                );
            }
        }
    }

    #[test]
    fn test_elevated_add_with_lambda_256_vectors() {
        // Build a dictionary from the first 8 vectors of Lambda_256.
        // This uses the predicate-enumerated points as a realistic test.
        let all_256 = enumerate_lambda_256();
        assert!(all_256.len() >= 8);

        let pairs: Vec<(usize, LatticeVector)> = all_256[..8]
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();

        let dict = EncodingDictionary::try_from_pairs(8, &pairs).unwrap();
        let stats = dict.elevated_addition_stats();

        eprintln!("Lambda_256 first-8 dictionary stats:");
        eprintln!(
            "  total_pairs={}, in_codebook={}, out_of_codebook={}, out_of_bounds={}",
            stats.total_pairs, stats.in_codebook, stats.out_of_codebook, stats.out_of_bounds
        );
        eprintln!(
            "  closure_rate={:.4}, commutative={}, identities={}",
            stats.closure_rate, stats.is_commutative, stats.identity_count
        );

        assert_eq!(stats.total_pairs, 64, "8x8 = 64 pairs");
        assert!(stats.is_commutative);
        // With vectors from the deep negative corner, most sums will overflow
        assert!(
            stats.out_of_bounds > 0,
            "Some sums should overflow trinary range"
        );
    }

    #[test]
    fn test_elevated_add_zero_vector() {
        // If the dictionary contains the zero vector [0,0,0,0,0,0,0,0],
        // it should act as identity for lattice addition.
        // Note: [0,0,...,0] IS in base_universe (sum=0 even, weight=0 even),
        // and it IS in Lambda_256 (prefix is all 0, but l_0 != 1, l_0 = 0).
        // Actually: is_in_base_universe requires l_0 != 1, and 0 != 1, so OK.
        // But is_in_lambda_2048: forbidden (0,1,1) -- only if l_0=0 AND l_1=1 AND l_2=1.
        // Zero vec has l_1=0, so no forbidden prefix fires.
        // is_in_lambda_1024: requires l_0=-1. Zero vec has l_0=0. FAILS.
        // So the zero vector is NOT in Lambda_256 (it fails at Lambda_1024).
        //
        // This means there's no additive identity in the Lambda_256 codebook.
        // Verify this:
        let all_256 = enumerate_lambda_256();
        let zero_present = all_256.iter().any(|v| v.iter().all(|&x| x == 0));
        assert!(
            !zero_present,
            "Zero vector is NOT in Lambda_256 (fails l_0=-1 for Lambda_1024)"
        );
    }

    // ================================================================
    // Layer 2: F_3 elevated addition tests
    // ================================================================

    #[test]
    fn test_lattice_add_f3_basic() {
        // F_3 wrapping: -1 + -1 = 1 (mod 3)
        let a: LatticeVector = [-1, -1, 0, 0, 0, 0, 0, 0];
        let b: LatticeVector = [-1, -1, 0, 0, 0, 0, 0, 0];
        let sum = lattice_add_f3(&a, &b);
        assert_eq!(sum[0], 1, "-1 + -1 = 1 in F_3");
        assert_eq!(sum[1], 1, "-1 + -1 = 1 in F_3");
    }

    #[test]
    fn test_lattice_add_f3_identity() {
        // Zero is the identity in F_3
        let a: LatticeVector = [-1, 1, 0, -1, 1, 0, -1, 1];
        let zero: LatticeVector = [0, 0, 0, 0, 0, 0, 0, 0];
        assert_eq!(lattice_add_f3(&a, &zero), a);
        assert_eq!(lattice_add_f3(&zero, &a), a);
    }

    #[test]
    fn test_lattice_add_f3_inverse() {
        // In F_3: the additive inverse of x is -x (i.e., 2x mod 3).
        // -(-1) = 1, -(0) = 0, -(1) = -1
        let a: LatticeVector = [-1, 1, 0, -1, 1, 0, -1, 1];
        let neg_a: LatticeVector = [1, -1, 0, 1, -1, 0, 1, -1];
        let sum = lattice_add_f3(&a, &neg_a);
        assert_eq!(sum, [0, 0, 0, 0, 0, 0, 0, 0], "a + (-a) = 0 in F_3");
    }

    #[test]
    fn test_f3_associativity_on_raw_vectors() {
        // F_3 addition is inherently associative (it's a group).
        let a: LatticeVector = [-1, 1, 0, -1, 1, 0, -1, 1];
        let b: LatticeVector = [1, 1, -1, 0, 0, -1, -1, 0];
        let c: LatticeVector = [0, -1, 1, 1, -1, 0, 0, -1];

        let ab = lattice_add_f3(&a, &b);
        let abc_left = lattice_add_f3(&ab, &c);

        let bc = lattice_add_f3(&b, &c);
        let abc_right = lattice_add_f3(&a, &bc);

        assert_eq!(
            abc_left, abc_right,
            "F_3 addition is associative on raw vectors"
        );
    }

    #[test]
    fn test_elevated_add_f3_synthetic() {
        let dict = make_test_dictionary_4();
        // F_3: Phi(0) +_3 Phi(0) = (-1,-1,-1,-1,0,0,0,0) +_3 same
        //      = (1,1,1,1,0,0,0,0) -- this is trinary but may not be in dict
        let r = dict.elevated_add_f3(0, 0).unwrap();
        match &r {
            ElevatedResultF3::InCodebook { sum_vec, .. } => {
                assert_eq!(*sum_vec, [1, 1, 1, 1, 0, 0, 0, 0]);
            }
            ElevatedResultF3::OutOfCodebook { sum_vec } => {
                assert_eq!(*sum_vec, [1, 1, 1, 1, 0, 0, 0, 0]);
            }
        }
    }

    #[test]
    fn test_elevated_add_f3_commutativity() {
        let dict = make_test_dictionary_4();
        for a in 0..4 {
            for b in 0..4 {
                let r_ab = dict.elevated_add_f3(a, b).unwrap();
                let r_ba = dict.elevated_add_f3(b, a).unwrap();
                assert_eq!(
                    r_ab, r_ba,
                    "F_3 elevated addition should be commutative: ({}, {})",
                    a, b
                );
            }
        }
    }

    #[test]
    fn test_elevated_addition_stats_f3_synthetic() {
        let dict = make_test_dictionary_4();
        let stats = dict.elevated_addition_stats_f3();

        eprintln!("F_3 synthetic 4-element dictionary stats:");
        eprintln!(
            "  total_pairs={}, in_codebook={}, out_of_codebook={}",
            stats.total_pairs, stats.in_codebook, stats.out_of_codebook
        );
        eprintln!(
            "  closure_rate={:.3}, commutative={}, identities={}",
            stats.closure_rate, stats.is_commutative, stats.identity_count
        );
        eprintln!(
            "  associativity_rate={:.3} ({} triples tested)",
            stats.associativity_rate, stats.associativity_triples_tested
        );

        assert_eq!(stats.total_pairs, 16);
        assert!(stats.is_commutative);
        assert_eq!(
            stats.in_codebook + stats.out_of_codebook,
            16,
            "No out-of-bounds in F_3"
        );
        // F_3 on raw vectors is always associative
        assert!(
            (stats.associativity_rate - 1.0).abs() < 1e-10,
            "F_3 is inherently associative"
        );
    }

    #[test]
    fn test_elevated_add_f3_lambda_256() {
        // Test F_3 addition on a realistic Lambda_256 sub-dictionary.
        let all_256 = enumerate_lambda_256();

        let pairs: Vec<(usize, LatticeVector)> = all_256[..16]
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();

        let dict = EncodingDictionary::try_from_pairs(16, &pairs).unwrap();
        let stats = dict.elevated_addition_stats_f3();

        eprintln!("F_3 Lambda_256 first-16 dictionary stats:");
        eprintln!(
            "  total_pairs={}, in_codebook={}, out_of_codebook={}",
            stats.total_pairs, stats.in_codebook, stats.out_of_codebook
        );
        eprintln!(
            "  closure_rate={:.4}, commutative={}, identities={}",
            stats.closure_rate, stats.is_commutative, stats.identity_count
        );
        eprintln!(
            "  associativity_rate={:.4} ({} triples tested)",
            stats.associativity_rate, stats.associativity_triples_tested
        );

        assert_eq!(stats.total_pairs, 256, "16x16 = 256 pairs");
        assert!(stats.is_commutative);
        assert!(
            (stats.associativity_rate - 1.0).abs() < 1e-10,
            "F_3 is always associative"
        );
        // The closure rate should be > 0 (F_3 wraps instead of overflowing)
        eprintln!(
            "  F_3 closure rate: {:.1}% ({}/{})",
            stats.closure_rate * 100.0,
            stats.in_codebook,
            stats.total_pairs
        );
    }

    #[test]
    fn test_elevated_diff_lambda_256() {
        // Z-difference on Lambda_256 sub-dictionary.
        // Since all l_0 = -1, difference gives l_0 = 0 (always in trinary range).
        let all_256 = enumerate_lambda_256();

        let pairs: Vec<(usize, LatticeVector)> = all_256[..8]
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();

        let dict = EncodingDictionary::try_from_pairs(8, &pairs).unwrap();

        let mut in_codebook = 0usize;
        let mut out_of_codebook = 0usize;
        let mut out_of_bounds = 0usize;

        for a in 0..8 {
            for b in 0..8 {
                match dict.elevated_diff(a, b).unwrap() {
                    ElevatedResult::InCodebook { .. } => in_codebook += 1,
                    ElevatedResult::OutOfCodebook { .. } => out_of_codebook += 1,
                    ElevatedResult::OutOfBounds { .. } => out_of_bounds += 1,
                }
            }
        }

        eprintln!(
            "Z-difference Lambda_256 first-8: in_codebook={}, out_of_codebook={}, out_of_bounds={}",
            in_codebook, out_of_codebook, out_of_bounds
        );

        // The self-difference a - a = 0 always, but 0 is NOT in Lambda_256.
        // So self-differences should all be OutOfCodebook.
        for a in 0..8 {
            let r = dict.elevated_diff(a, a).unwrap();
            assert!(
                matches!(r, ElevatedResult::OutOfCodebook { sum_vec }
                if sum_vec == [0, 0, 0, 0, 0, 0, 0, 0]),
                "a - a = 0, which is not in Lambda_256"
            );
        }
    }

    // ================================================================
    // Multiplication Coupling Tests (Thesis D, C-466)
    // ================================================================

    /// Helper: build a dim=16 dictionary from the first 16 Lambda_256 vectors
    /// and the corresponding multiplication table.
    fn sedenion_coupling_setup() -> (EncodingDictionary, cd_kernel::mult_table::CdMultTable) {
        let lambda = enumerate_lambda_256();
        assert!(lambda.len() >= 16);
        let pairs: Vec<(usize, LatticeVector)> = lambda[..16]
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        let dict = EncodingDictionary::try_from_pairs(16, &pairs).unwrap();
        let table = cd_kernel::mult_table::CdMultTable::generate(16);
        (dict, table)
    }

    #[test]
    fn test_multiplication_coupling_sedenion_basic() {
        let (dict, table) = sedenion_coupling_setup();
        let coupling = compute_multiplication_coupling(&dict, &table);

        assert_eq!(coupling.dim, 16);
        assert_eq!(coupling.results.len(), 16);
        assert!(
            coupling.rank > 0 && coupling.rank <= 8,
            "rank should be in [1,8], got {}",
            coupling.rank
        );

        // Basis 0 (e_0 = identity): multiplication is identity permutation,
        // so rho(0) is the identity map on the subspace.
        let r0 = &coupling.results[0];
        assert!(
            r0.unsigned_consistent,
            "e_0 * e_c = e_c, unsigned coupling must be consistent"
        );
        assert!(
            r0.signed_consistent,
            "e_0 * e_c = +1 * e_c, signed coupling must be consistent"
        );

        // det(I_r) = 1 in the reduced space
        let det_u = r0.unsigned_det.unwrap();
        assert!(
            (det_u - 1.0).abs() < 1e-6,
            "det(rho_unsigned(0)) should be 1, got {det_u}"
        );
        let det_s = r0.signed_det.unwrap();
        assert!(
            (det_s - 1.0).abs() < 1e-6,
            "det(rho_signed(0)) should be 1, got {det_s}"
        );

        // Report summary
        eprintln!("=== Sedenion (dim=16) Multiplication Coupling ===");
        eprintln!("Lattice vector rank: {}/8", coupling.rank);
        eprintln!(
            "Unsigned consistent: {}/16",
            coupling.unsigned_consistent_count
        );
        eprintln!(
            "Signed consistent:   {}/16",
            coupling.signed_consistent_count
        );
        eprintln!("Unsigned determinants: {:?}", coupling.unsigned_dets);
        eprintln!("Signed determinants:   {:?}", coupling.signed_dets);

        for r in &coupling.results {
            eprintln!(
                "  b={:2}: u_ok={} u_det={:?} u_res={:.2e} | s_ok={} s_det={:?} s_res={:.2e}",
                r.basis_index,
                if r.unsigned_consistent { "Y" } else { "N" },
                r.unsigned_det,
                r.unsigned_max_residual,
                if r.signed_consistent { "Y" } else { "N" },
                r.signed_det,
                r.signed_max_residual,
            );
        }
    }

    #[test]
    fn test_multiplication_coupling_identity_element() {
        let (dict, table) = sedenion_coupling_setup();
        let coupling = compute_multiplication_coupling(&dict, &table);

        let r0 = &coupling.results[0];
        assert!(r0.unsigned_consistent);
        assert!(r0.signed_consistent);
        assert!((r0.unsigned_det.unwrap() - 1.0).abs() < 1e-6);
        assert!((r0.signed_det.unwrap() - 1.0).abs() < 1e-6);
        assert!(r0.unsigned_max_residual < 1e-10);
        assert!(r0.signed_max_residual < 1e-10);
    }

    #[test]
    fn test_multiplication_coupling_sedenion_characterize() {
        let (dict, table) = sedenion_coupling_setup();
        let coupling = compute_multiplication_coupling(&dict, &table);

        let unsigned_bases: Vec<usize> = coupling
            .results
            .iter()
            .filter(|r| r.unsigned_consistent)
            .map(|r| r.basis_index)
            .collect();
        let signed_bases: Vec<usize> = coupling
            .results
            .iter()
            .filter(|r| r.signed_consistent)
            .map(|r| r.basis_index)
            .collect();

        // At minimum, basis 0 must always work
        assert!(unsigned_bases.contains(&0));
        assert!(signed_bases.contains(&0));

        eprintln!("Rank: {}", coupling.rank);
        eprintln!("Unsigned consistent bases: {:?}", unsigned_bases);
        eprintln!("Signed consistent bases:   {:?}", signed_bases);
        eprintln!("Unsigned dets: {:?}", coupling.unsigned_dets);
        eprintln!("Signed dets:   {:?}", coupling.signed_dets);

        // The key research question: how many basis elements have
        // consistent linear coupling? Is it all of them, or a subset?
        // Record the answer for C-466.
        eprintln!(
            "C-466 result: {}/{} unsigned, {}/{} signed",
            coupling.unsigned_consistent_count,
            coupling.dim,
            coupling.signed_consistent_count,
            coupling.dim
        );
    }

    #[test]
    fn test_multiplication_coupling_pathion() {
        let lambda = enumerate_lambda_256();
        assert!(
            lambda.len() >= 32,
            "Lambda_256 has {} vectors, need 32",
            lambda.len()
        );
        let pairs: Vec<(usize, LatticeVector)> = lambda[..32]
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        let dict = EncodingDictionary::try_from_pairs(32, &pairs).unwrap();
        let table = cd_kernel::mult_table::CdMultTable::generate(32);

        let coupling = compute_multiplication_coupling(&dict, &table);

        assert_eq!(coupling.dim, 32);
        assert_eq!(coupling.results.len(), 32);

        // Identity element
        let r0 = &coupling.results[0];
        assert!(r0.unsigned_consistent, "rho(0) must be identity for dim=32");
        assert!((r0.unsigned_det.unwrap() - 1.0).abs() < 1e-6);

        eprintln!("=== Pathion (dim=32) Multiplication Coupling ===");
        eprintln!("Rank: {}/8", coupling.rank);
        eprintln!(
            "Unsigned consistent: {}/32",
            coupling.unsigned_consistent_count
        );
        eprintln!(
            "Signed consistent:   {}/32",
            coupling.signed_consistent_count
        );
    }

    #[test]
    fn test_gram_schmidt_basis_rank() {
        // Verify that gram_schmidt_basis correctly determines rank.
        let lambda = enumerate_lambda_256();
        let phi16: Vec<[f64; 8]> = lambda[..16]
            .iter()
            .map(|lv| {
                let mut f = [0.0f64; 8];
                for (fv, &iv) in f.iter_mut().zip(lv.iter()) {
                    *fv = iv as f64;
                }
                f
            })
            .collect();
        let (basis16, rank16) = gram_schmidt_basis(&phi16);
        assert_eq!(basis16.len(), rank16);
        assert!(
            (1..=8).contains(&rank16),
            "rank should be in [1,8], got {rank16}"
        );

        // Full Lambda_256 should span more dimensions
        let phi_all: Vec<[f64; 8]> = lambda
            .iter()
            .map(|lv| {
                let mut f = [0.0f64; 8];
                for (fv, &iv) in f.iter_mut().zip(lv.iter()) {
                    *fv = iv as f64;
                }
                f
            })
            .collect();
        let (_, rank_all) = gram_schmidt_basis(&phi_all);
        assert!(
            rank_all >= rank16,
            "full Lambda_256 rank ({rank_all}) must be >= dim=16 rank ({rank16})"
        );

        eprintln!("Rank of first 16 Lambda_256 vectors: {rank16}");
        eprintln!(
            "Rank of all {} Lambda_256 vectors: {rank_all}",
            lambda.len()
        );
    }

    #[test]
    fn test_invert_nxn_identity() {
        for n in [3, 5, 8] {
            let m: Vec<Vec<f64>> = (0..n)
                .map(|i| {
                    let mut row = vec![0.0; n];
                    row[i] = 1.0;
                    row
                })
                .collect();
            let inv = invert_nxn(&m, n).unwrap();
            for (i, row) in inv.iter().enumerate() {
                for (j, &val) in row.iter().enumerate() {
                    let expected = if i == j { 1.0 } else { 0.0 };
                    assert!(
                        (val - expected).abs() < 1e-12,
                        "I^-1 [{i}][{j}] = {val} for n={n}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_det_nxn_identity() {
        for n in [3, 5, 8] {
            let m: Vec<Vec<f64>> = (0..n)
                .map(|i| {
                    let mut row = vec![0.0; n];
                    row[i] = 1.0;
                    row
                })
                .collect();
            assert!(
                (det_nxn(&m, n) - 1.0).abs() < 1e-12,
                "det(I_{n}) should be 1"
            );
        }
    }

    // ================================================================
    // Coset Obstruction and Affine Closure Analysis (C-498, C-499)
    // ================================================================

    #[test]
    fn test_coset_obstruction_and_affine_closure() {
        use std::collections::HashSet;

        // ---- Phase 1: Filtration level enumeration ----
        let base_universe = enumerate_lattice_by_predicate(is_in_base_universe);
        let lambda_2048 = enumerate_lattice_by_predicate(is_in_lambda_2048);
        let lambda_1024 = enumerate_lattice_by_predicate(is_in_lambda_1024);
        let lambda_512 = enumerate_lattice_by_predicate(is_in_lambda_512);
        let lambda_256 = enumerate_lambda_256();

        eprintln!("=== Filtration Level Sizes ===");
        eprintln!("  Base universe: {}", base_universe.len());
        eprintln!("  Lambda_2048:   {}", lambda_2048.len());
        eprintln!("  Lambda_1024:   {}", lambda_1024.len());
        eprintln!("  Lambda_512:    {}", lambda_512.len());
        eprintln!("  Lambda_256:    {}", lambda_256.len());

        // Strict inclusion chain
        assert!(lambda_256.len() < lambda_512.len());
        assert!(lambda_512.len() < lambda_1024.len());
        assert!(lambda_1024.len() < lambda_2048.len());
        assert!(lambda_2048.len() < base_universe.len());

        // ---- Phase 2: Coset decomposition of Lambda_2048 by l_0 ----
        let l0_neg1: Vec<&LatticeVector> = lambda_2048.iter().filter(|v| v[0] == -1).collect();
        let l0_zero: Vec<&LatticeVector> = lambda_2048.iter().filter(|v| v[0] == 0).collect();
        let l0_pos1: Vec<&LatticeVector> = lambda_2048.iter().filter(|v| v[0] == 1).collect();

        eprintln!("\n=== Lambda_2048 Coset Decomposition by l_0 ===");
        eprintln!("  l_0 = -1: {} vectors", l0_neg1.len());
        eprintln!("  l_0 =  0: {} vectors", l0_zero.len());
        eprintln!(
            "  l_0 = +1: {} vectors (excluded by base universe)",
            l0_pos1.len()
        );

        assert_eq!(l0_pos1.len(), 0, "l_0 = +1 excluded from base universe");
        assert_eq!(
            l0_neg1.len() + l0_zero.len(),
            lambda_2048.len(),
            "Coset partition exhaustive"
        );

        // Lambda_1024 is exactly the l_0=-1 sub-lattice of Lambda_2048
        // (minus further exclusions). All l0_neg1 satisfy is_in_lambda_2048 but
        // not all satisfy is_in_lambda_1024 (additional exclusions).
        assert!(lambda_1024.len() <= l0_neg1.len());
        assert!(
            lambda_1024.iter().all(|v| v[0] == -1),
            "Lambda_1024 requires l_0 = -1"
        );

        // ---- Phase 3: Prove the coset obstruction ----
        // All Lambda_256 vectors have l_0 = -1
        assert!(
            lambda_256.iter().all(|v| v[0] == -1),
            "All Lambda_256 vectors have l_0 = -1"
        );

        // Z-addition: (-1) + (-1) = -2, leaving {-1,0,1}^8
        let a0 = &lambda_256[0];
        let b0 = &lambda_256[1];
        let z_sum = lattice_add(a0, b0);
        assert_eq!(
            z_sum[0], -2,
            "Z-addition: l_0 = (-1)+(-1) = -2 (out of bounds)"
        );

        // F_3-addition: (-1) + (-1) = 1, landing in forbidden coset
        let f3_sum = lattice_add_f3(a0, b0);
        assert_eq!(f3_sum[0], 1, "F_3: l_0 = (-1)+(-1) = 1");
        assert!(
            !is_in_base_universe(&f3_sum),
            "l_0=1 excluded from base universe"
        );

        // Exhaustive: ALL pairs of Lambda_256 under F_3-addition give l_0=1
        for a in &lambda_256 {
            for b in &lambda_256 {
                let s = lattice_add_f3(a, b);
                assert_eq!(s[0], 1, "Every F_3 sum of Lambda_256 vectors has l_0 = 1");
            }
        }

        eprintln!("\n=== Phase 3: Coset Obstruction Confirmed ===");
        eprintln!("  Z-addition:  l_0 = -2 (out of bounds) for ALL pairs");
        eprintln!("  F_3-addition: l_0 = +1 (forbidden coset) for ALL pairs");

        // ---- Phase 4: F_3 closure of the l_0=0 sub-lattice of Lambda_2048 ----
        let l0_zero_set: HashSet<LatticeVector> = l0_zero.iter().copied().cloned().collect();
        let n_zero = l0_zero.len();
        let mut zero_closure_count = 0usize;
        let zero_total = n_zero * n_zero;

        for a in &l0_zero {
            for b in &l0_zero {
                let s = lattice_add_f3(a, b);
                // In F_3: l_0 = 0+0 = 0, so sum stays in l_0=0 coset
                assert_eq!(s[0], 0, "F_3: 0+0 = 0 for l_0 coordinate");
                if l0_zero_set.contains(&s) {
                    zero_closure_count += 1;
                }
            }
        }
        let zero_closure_rate = zero_closure_count as f64 / zero_total as f64;

        eprintln!("\n=== Phase 4: l_0=0 Sub-lattice F_3 Closure ===");
        eprintln!(
            "  {}/{} pairs closed = {:.4} ({:.1}%)",
            zero_closure_count,
            zero_total,
            zero_closure_rate,
            zero_closure_rate * 100.0
        );
        // The l_0=0 sub-lattice should have positive closure (it's an actual subgroup)
        assert!(
            zero_closure_count > 0,
            "l_0=0 sub-lattice must have some closure under F_3"
        );

        // ---- Phase 5: Affine F_3 closure on Lambda_256 ----
        // Operation: a +_3 b -_3 p, where p is a fixed base point.
        // This maps l_0: (-1)+(-1)-(-1) = (-1)+(-1)+(+1) = 1+1 = -1 in F_3.
        let lambda_256_set: HashSet<LatticeVector> = lambda_256.iter().copied().collect();
        let n_256 = lambda_256.len();
        let total_256 = n_256 * n_256;

        // Verify l_0 is preserved by affine operation
        let base_point = lambda_256[0];
        let neg_base = lattice_negate_f3(&base_point);
        let test_affine = lattice_add_f3(&lattice_add_f3(a0, b0), &neg_base);
        assert_eq!(test_affine[0], -1, "Affine F_3: l_0 = (-1)+(-1)-(-1) = -1");

        // Full sweep with base point = Lambda_256[0]
        let mut affine_closure_count = 0usize;
        for a in &lambda_256 {
            for b in &lambda_256 {
                let ab = lattice_add_f3(a, b);
                let result = lattice_add_f3(&ab, &neg_base);
                // Verify l_0 preservation for every pair
                assert_eq!(result[0], -1, "Affine sum preserves l_0 = -1");
                if lambda_256_set.contains(&result) {
                    affine_closure_count += 1;
                }
            }
        }
        let affine_rate = affine_closure_count as f64 / total_256 as f64;

        eprintln!("\n=== Phase 5: Affine F_3 Closure on Lambda_256 ===");
        eprintln!("  base = Lambda_256[0] = {:?}", base_point);
        eprintln!(
            "  {}/{} pairs closed = {:.4} ({:.1}%)",
            affine_closure_count,
            total_256,
            affine_rate,
            affine_rate * 100.0
        );

        // ---- Phase 6: Test multiple base points for rate variation ----
        let test_bases = [0, n_256 / 4, n_256 / 2, n_256 - 1];
        let mut rates = Vec::new();
        for &idx in &test_bases {
            if idx >= n_256 {
                continue;
            }
            let bp = lambda_256[idx];
            let nbp = lattice_negate_f3(&bp);
            let mut count = 0usize;
            for a in &lambda_256 {
                for b in &lambda_256 {
                    let ab = lattice_add_f3(a, b);
                    let result = lattice_add_f3(&ab, &nbp);
                    if lambda_256_set.contains(&result) {
                        count += 1;
                    }
                }
            }
            let rate = count as f64 / total_256 as f64;
            rates.push((idx, count, rate));
            eprintln!(
                "  base[{}] = {:?}: {}/{} = {:.1}%",
                idx,
                bp,
                count,
                total_256,
                rate * 100.0
            );
        }

        // ---- Phase 7: Affine F_3 closure at Lambda_512 and Lambda_1024 ----
        for (name, level) in [("Lambda_512", &lambda_512), ("Lambda_1024", &lambda_1024)] {
            let level_set: HashSet<LatticeVector> = level.iter().copied().collect();
            let n = level.len();
            let total = n * n;
            let bp = level[0];
            let nbp = lattice_negate_f3(&bp);
            let mut count = 0usize;
            for a in level.iter() {
                for b in level.iter() {
                    let ab = lattice_add_f3(a, b);
                    let result = lattice_add_f3(&ab, &nbp);
                    if level_set.contains(&result) {
                        count += 1;
                    }
                }
            }
            let rate = count as f64 / total as f64;
            eprintln!(
                "\n  Affine F_3 on {} ({} vectors): {}/{} = {:.1}%",
                name,
                n,
                count,
                total,
                rate * 100.0
            );
        }

        // ---- Summary ----
        eprintln!("\n=== COSET ANALYSIS SUMMARY ===");
        eprintln!("C-498: Coset Obstruction -- Lambda_256 has 0% Z/F_3 closure");
        eprintln!("  because l_0=-1 coset maps to l_0=+1 (forbidden) under addition.");
        eprintln!(
            "  l_0=0 sub-lattice of Lambda_2048 has {:.1}% F_3 closure (subgroup).",
            zero_closure_rate * 100.0
        );
        eprintln!(
            "  Affine F_3 on Lambda_256: {:.1}% closure (coset-corrected).",
            affine_rate * 100.0
        );
    }

    /// Systematic base-point sweep for affine F_3 closure on Lambda_256.
    ///
    /// Tests ALL 256 base points to characterize how closure rate varies,
    /// then correlates closure with lattice properties (Hamming weight,
    /// coordinate pattern, filtration depth).
    #[test]
    #[ignore = "heavy research lane: exhaustive affine F_3 base-point sweep"]
    fn test_affine_f3_closure_full_basepoint_sweep() {
        use std::collections::HashSet;

        let lambda_256 = enumerate_lambda_256();
        let n = lambda_256.len();
        let total_pairs = n * n;
        let lambda_set: HashSet<LatticeVector> = lambda_256.iter().copied().collect();

        eprintln!(
            "\n=== Affine F_3 Closure: Full Base-Point Sweep on Lambda_256 ({} vectors) ===",
            n
        );

        // Compute closure rate for every base point
        let mut rates: Vec<(usize, f64, LatticeVector)> = Vec::with_capacity(n);
        for (idx, bp) in lambda_256.iter().enumerate() {
            let nbp = lattice_negate_f3(bp);
            let mut count = 0usize;
            for a in &lambda_256 {
                for b in &lambda_256 {
                    let ab = lattice_add_f3(a, b);
                    let result = lattice_add_f3(&ab, &nbp);
                    if lambda_set.contains(&result) {
                        count += 1;
                    }
                }
            }
            let rate = count as f64 / total_pairs as f64;
            rates.push((idx, rate, *bp));
        }

        // Sort by rate to find extremes
        rates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let min_rate = rates[0].1;
        let max_rate = rates[n - 1].1;
        let mean_rate = rates.iter().map(|r| r.1).sum::<f64>() / n as f64;
        let std_rate =
            (rates.iter().map(|r| (r.1 - mean_rate).powi(2)).sum::<f64>() / n as f64).sqrt();

        eprintln!("\n--- Rate Statistics ---");
        eprintln!(
            "  Min:  {:.4} ({:.1}%) at idx {} = {:?}",
            min_rate,
            min_rate * 100.0,
            rates[0].0,
            rates[0].2
        );
        eprintln!(
            "  Max:  {:.4} ({:.1}%) at idx {} = {:?}",
            max_rate,
            max_rate * 100.0,
            rates[n - 1].0,
            rates[n - 1].2
        );
        eprintln!("  Mean: {:.4} ({:.1}%)", mean_rate, mean_rate * 100.0);
        eprintln!("  Std:  {:.4}", std_rate);

        // Rate histogram (by 5% bucket)
        let mut histogram = [0usize; 20]; // 0-5%, 5-10%, ..., 95-100%
        for r in &rates {
            let bucket = (r.1 * 20.0).floor() as usize;
            let bucket = bucket.min(19);
            histogram[bucket] += 1;
        }
        eprintln!("\n--- Rate Histogram (5% buckets) ---");
        for (i, &count) in histogram.iter().enumerate() {
            if count > 0 {
                eprintln!("  {}-{}%: {} base points", i * 5, (i + 1) * 5, count);
            }
        }

        // Bottom 5 and top 5
        eprintln!("\n--- Bottom 5 ---");
        for &(idx, rate, ref v) in rates.iter().take(5) {
            let hw: usize = v.iter().filter(|&&x| x != 0).count();
            let cs: i32 = v.iter().map(|&x| x as i32).sum();
            eprintln!(
                "  idx={}: rate={:.4} ({:.1}%), hw={}, csum={}, v={:?}",
                idx,
                rate,
                rate * 100.0,
                hw,
                cs,
                v
            );
        }
        eprintln!("\n--- Top 5 ---");
        for &(idx, rate, ref v) in rates.iter().rev().take(5) {
            let hw: usize = v.iter().filter(|&&x| x != 0).count();
            let cs: i32 = v.iter().map(|&x| x as i32).sum();
            eprintln!(
                "  idx={}: rate={:.4} ({:.1}%), hw={}, csum={}, v={:?}",
                idx,
                rate,
                rate * 100.0,
                hw,
                cs,
                v
            );
        }

        // Correlations: Hamming weight vs rate, coordinate sum vs rate
        let hamming_weights: Vec<usize> = lambda_256
            .iter()
            .map(|v| v.iter().filter(|&&x| x != 0).count())
            .collect();
        let coord_sums: Vec<i32> = lambda_256
            .iter()
            .map(|v| v.iter().map(|&x| x as i32).sum())
            .collect();

        // Build sorted-by-index rates for correlation
        let mut rates_by_idx = vec![0.0f64; n];
        for &(idx, rate, _) in &rates {
            rates_by_idx[idx] = rate;
        }

        // Spearman rank correlation (approximate via Pearson on ranks)
        let hw_corr = pearson_correlation(
            &hamming_weights
                .iter()
                .map(|&w| w as f64)
                .collect::<Vec<_>>(),
            &rates_by_idx,
        );
        let cs_corr = pearson_correlation(
            &coord_sums.iter().map(|&s| s as f64).collect::<Vec<_>>(),
            &rates_by_idx,
        );

        eprintln!("\n--- Correlations ---");
        eprintln!("  Hamming weight vs rate: r = {:.4}", hw_corr);
        eprintln!("  Coordinate sum vs rate: r = {:.4}", cs_corr);

        // Mean rate grouped by Hamming weight
        let mut hw_groups: std::collections::HashMap<usize, Vec<f64>> =
            std::collections::HashMap::new();
        for (i, &hw) in hamming_weights.iter().enumerate() {
            hw_groups.entry(hw).or_default().push(rates_by_idx[i]);
        }
        let mut hw_keys: Vec<usize> = hw_groups.keys().copied().collect();
        hw_keys.sort();
        eprintln!("\n--- Mean Rate by Hamming Weight ---");
        for hw in hw_keys {
            let group = &hw_groups[&hw];
            let mean = group.iter().sum::<f64>() / group.len() as f64;
            eprintln!(
                "  hw={}: mean rate={:.4} ({:.1}%), n={}",
                hw,
                mean,
                mean * 100.0,
                group.len()
            );
        }

        // Assertions
        assert!(min_rate > 0.0, "Some closure must exist");
        assert!(max_rate < 1.0, "100% closure would mean affine subgroup");
        assert!(min_rate >= 0.20, "Min rate should be at least 20%");
        assert!(max_rate <= 0.50, "Max rate should be at most 50%");
    }

    /// Affine F_3 closure across filtration levels with representative base points.
    ///
    /// At Lambda_512 (512 vectors) and Lambda_1024 (1026 vectors), a full sweep
    /// is expensive. We sample 20 base points spread evenly and compute closure
    /// rate for each, comparing against Lambda_256's known range.
    #[test]
    fn test_affine_f3_closure_across_levels() {
        use std::collections::HashSet;

        let lambda_256 = enumerate_lambda_256();
        let lambda_512 = enumerate_lattice_by_predicate(is_in_lambda_512);
        let lambda_1024 = enumerate_lattice_by_predicate(is_in_lambda_1024);

        let n_sample = 20;

        for (name, level) in [
            ("Lambda_256", &lambda_256),
            ("Lambda_512", &lambda_512),
            ("Lambda_1024", &lambda_1024),
        ] {
            let n = level.len();
            let total_pairs = n * n;
            let level_set: HashSet<LatticeVector> = level.iter().copied().collect();

            // Sample n_sample base points evenly spread
            let step = (n / n_sample).max(1);
            let mut rates = Vec::new();

            for i in 0..n_sample {
                let idx = (i * step).min(n - 1);
                let bp = &level[idx];
                let nbp = lattice_negate_f3(bp);
                let mut count = 0usize;
                for a in level.iter() {
                    for b in level.iter() {
                        let ab = lattice_add_f3(a, b);
                        let result = lattice_add_f3(&ab, &nbp);
                        if level_set.contains(&result) {
                            count += 1;
                        }
                    }
                }
                let rate = count as f64 / total_pairs as f64;
                rates.push(rate);
            }

            let mean = rates.iter().sum::<f64>() / rates.len() as f64;
            let min = rates.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = rates.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

            eprintln!(
                "{} ({} vectors): mean={:.4} ({:.1}%), range=[{:.4}, {:.4}]",
                name,
                n,
                mean,
                mean * 100.0,
                min,
                max
            );

            // All levels should have some closure (>20%) but not full (<60%)
            assert!(mean > 0.20, "{} mean closure should exceed 20%", name);
            assert!(mean < 0.60, "{} mean closure should be below 60%", name);
        }
    }

    /// Full base-point sweep of affine F_3 closure on Lambda_512.
    ///
    /// Lambda_512 showed anomalous wide variance (24-34%) in the sampled test.
    /// This test sweeps ALL 512 base points to characterize the full distribution,
    /// and correlates with lattice properties to explain the variance.
    #[test]
    #[ignore = "heavy research lane: exhaustive affine F_3 Lambda_512 sweep"]
    fn test_affine_f3_closure_lambda512_full_sweep() {
        use std::collections::HashSet;

        let lambda_512 = enumerate_lattice_by_predicate(is_in_lambda_512);
        let n = lambda_512.len();
        assert_eq!(n, 512);
        let total_pairs = n * n;
        let lambda_set: HashSet<LatticeVector> = lambda_512.iter().copied().collect();

        eprintln!(
            "\n=== Affine F_3 Closure: Full Base-Point Sweep on Lambda_512 ({} vectors) ===",
            n
        );

        let mut rates: Vec<(usize, f64, LatticeVector)> = Vec::with_capacity(n);
        for (idx, bp) in lambda_512.iter().enumerate() {
            let nbp = lattice_negate_f3(bp);
            let mut count = 0usize;
            for a in &lambda_512 {
                for b in &lambda_512 {
                    let ab = lattice_add_f3(a, b);
                    let result = lattice_add_f3(&ab, &nbp);
                    if lambda_set.contains(&result) {
                        count += 1;
                    }
                }
            }
            let rate = count as f64 / total_pairs as f64;
            rates.push((idx, rate, *bp));
        }

        rates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let min_rate = rates[0].1;
        let max_rate = rates[n - 1].1;
        let mean_rate = rates.iter().map(|r| r.1).sum::<f64>() / n as f64;
        let std_rate =
            (rates.iter().map(|r| (r.1 - mean_rate).powi(2)).sum::<f64>() / n as f64).sqrt();

        eprintln!("\n--- Rate Statistics ---");
        eprintln!("  Min:  {:.4} ({:.1}%)", min_rate, min_rate * 100.0);
        eprintln!("  Max:  {:.4} ({:.1}%)", max_rate, max_rate * 100.0);
        eprintln!("  Mean: {:.4} ({:.1}%)", mean_rate, mean_rate * 100.0);
        eprintln!("  Std:  {:.4}", std_rate);

        // Rate histogram
        let mut histogram = [0usize; 20];
        for r in &rates {
            let bucket = (r.1 * 20.0).floor() as usize;
            histogram[bucket.min(19)] += 1;
        }
        eprintln!("\n--- Rate Histogram (5% buckets) ---");
        for (i, &count) in histogram.iter().enumerate() {
            if count > 0 {
                eprintln!("  {}-{}%: {} base points", i * 5, (i + 1) * 5, count);
            }
        }

        // Bottom 5 and top 5
        eprintln!("\n--- Bottom 5 ---");
        for &(idx, rate, ref v) in rates.iter().take(5) {
            let hw: usize = v.iter().filter(|&&x| x != 0).count();
            eprintln!(
                "  idx={}: rate={:.4} ({:.1}%), hw={}, v={:?}",
                idx,
                rate,
                rate * 100.0,
                hw,
                v
            );
        }
        eprintln!("\n--- Top 5 ---");
        for &(idx, rate, ref v) in rates.iter().rev().take(5) {
            let hw: usize = v.iter().filter(|&&x| x != 0).count();
            eprintln!(
                "  idx={}: rate={:.4} ({:.1}%), hw={}, v={:?}",
                idx,
                rate,
                rate * 100.0,
                hw,
                v
            );
        }

        // Correlation with Hamming weight
        let mut rates_by_idx = vec![0.0f64; n];
        for &(idx, rate, _) in &rates {
            rates_by_idx[idx] = rate;
        }
        let hws: Vec<f64> = lambda_512
            .iter()
            .map(|v| v.iter().filter(|&&x| x != 0).count() as f64)
            .collect();
        let hw_corr = pearson_correlation(&hws, &rates_by_idx);

        // Correlation with l_1 value (is the C-501 contaminant related?)
        let l1_vals: Vec<f64> = lambda_512.iter().map(|v| v[1] as f64).collect();
        let l1_corr = pearson_correlation(&l1_vals, &rates_by_idx);

        // Correlation with number of +1 coordinates
        let plus1_counts: Vec<f64> = lambda_512
            .iter()
            .map(|v| v.iter().filter(|&&x| x == 1).count() as f64)
            .collect();
        let p1_corr = pearson_correlation(&plus1_counts, &rates_by_idx);

        eprintln!("\n--- Correlations ---");
        eprintln!("  Hamming weight vs rate:  r = {:.4}", hw_corr);
        eprintln!("  l_1 value vs rate:       r = {:.4}", l1_corr);
        eprintln!("  #(+1 coords) vs rate:    r = {:.4}", p1_corr);

        // Mean rate grouped by Hamming weight
        let mut hw_groups: std::collections::HashMap<usize, Vec<f64>> =
            std::collections::HashMap::new();
        for (i, v) in lambda_512.iter().enumerate() {
            let hw = v.iter().filter(|&&x| x != 0).count();
            hw_groups.entry(hw).or_default().push(rates_by_idx[i]);
        }
        let mut hw_keys: Vec<usize> = hw_groups.keys().copied().collect();
        hw_keys.sort();
        eprintln!("\n--- Mean Rate by Hamming Weight ---");
        for hw in hw_keys {
            let group = &hw_groups[&hw];
            let mean = group.iter().sum::<f64>() / group.len() as f64;
            let std =
                (group.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / group.len() as f64).sqrt();
            eprintln!(
                "  hw={}: mean={:.4} ({:.1}%), std={:.4}, n={}",
                hw,
                mean,
                mean * 100.0,
                std,
                group.len()
            );
        }

        // Mean rate grouped by l_1 value
        let mut l1_groups: std::collections::HashMap<i8, Vec<f64>> =
            std::collections::HashMap::new();
        for (i, v) in lambda_512.iter().enumerate() {
            l1_groups.entry(v[1]).or_default().push(rates_by_idx[i]);
        }
        eprintln!("\n--- Mean Rate by l_1 Value ---");
        for l1 in [-1i8, 0, 1] {
            if let Some(group) = l1_groups.get(&l1) {
                let mean = group.iter().sum::<f64>() / group.len() as f64;
                eprintln!(
                    "  l_1={}: mean={:.4} ({:.1}%), n={}",
                    l1,
                    mean,
                    mean * 100.0,
                    group.len()
                );
            }
        }

        assert!(min_rate > 0.15, "Min rate should be at least 15%");
        assert!(max_rate < 0.50, "Max rate should be at most 50%");
    }

    /// Pearson correlation coefficient for two f64 slices.
    fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
        let n = x.len() as f64;
        let mx = x.iter().sum::<f64>() / n;
        let my = y.iter().sum::<f64>() / n;
        let mut sxy = 0.0;
        let mut sxx = 0.0;
        let mut syy = 0.0;
        for (xi, yi) in x.iter().zip(y.iter()) {
            let dx = xi - mx;
            let dy = yi - my;
            sxy += dx * dy;
            sxx += dx * dx;
            syy += dy * dy;
        }
        if sxx < 1e-15 || syy < 1e-15 {
            return 0.0;
        }
        sxy / (sxx * syy).sqrt()
    }

    #[test]
    fn test_lattice_negate_f3_basic() {
        let v: LatticeVector = [-1, 0, 1, -1, 1, 0, -1, 1];
        let neg = lattice_negate_f3(&v);
        assert_eq!(neg, [1, 0, -1, 1, -1, 0, 1, -1]);

        // Double negation is identity
        let double_neg = lattice_negate_f3(&neg);
        assert_eq!(double_neg, v);

        // Negation of zero is zero
        let zero: LatticeVector = [0, 0, 0, 0, 0, 0, 0, 0];
        assert_eq!(lattice_negate_f3(&zero), zero);

        // a + (-a) = 0 in F_3
        let sum = lattice_add_f3(&v, &neg);
        assert_eq!(sum, [0, 0, 0, 0, 0, 0, 0, 0]);
    }

    // ================================================================
    // Phase A (T4): Lambda_4096 carrier tests
    // ================================================================

    #[test]
    fn test_lambda_4096_carrier_count() {
        // Lambda_4096 = base universe (no additional exclusions).
        // Should be a strict superset of Lambda_2048.
        let l4096 = enumerate_lambda_4096();
        let l2048 = enumerate_lattice_by_predicate(is_in_lambda_2048);
        let base = enumerate_lattice_by_predicate(is_in_base_universe);

        eprintln!(
            "Lambda_4096: {} vectors (= base universe: {}), Lambda_2048: {}",
            l4096.len(),
            base.len(),
            l2048.len()
        );

        // Lambda_4096 == base universe
        assert_eq!(
            l4096.len(),
            base.len(),
            "Lambda_4096 should equal base universe"
        );

        // Lambda_4096 > Lambda_2048 (strict superset)
        assert!(
            l4096.len() > l2048.len(),
            "Lambda_4096 ({}) must be a strict superset of Lambda_2048 ({})",
            l4096.len(),
            l2048.len()
        );

        // Every Lambda_2048 vector should be in Lambda_4096
        let l4096_set: std::collections::HashSet<LatticeVector> = l4096.iter().copied().collect();
        for v in &l2048 {
            assert!(
                l4096_set.contains(v),
                "Lambda_2048 vector {:?} not in Lambda_4096",
                v
            );
        }
    }

    #[test]
    fn test_lambda_4096_parity_constraints() {
        // Verify all 4 octonion parity laws hold for Lambda_4096.
        let l4096 = enumerate_lambda_4096();
        let (n, n_tri, n_sum, n_wt, n_l0, all_pass) = verify_octonion_parity_constraints(&l4096);

        eprintln!("Lambda_4096 parity check: n={n}");
        eprintln!("  trinary: {n_tri}/{n}");
        eprintln!("  even_sum: {n_sum}/{n}");
        eprintln!("  even_weight: {n_wt}/{n}");
        eprintln!("  l_0 != +1: {n_l0}/{n}");

        assert!(
            all_pass,
            "All 4 octonion parity constraints must hold for Lambda_4096. \
             trinary={n_tri}/{n}, even_sum={n_sum}/{n}, even_weight={n_wt}/{n}, l0={n_l0}/{n}"
        );
    }

    #[test]
    fn test_octonion_parity_proof_dim4096() {
        // Cross-validate: parity constraints hold for EVERY filtration level.
        // This is the algebraic proof that octonion structure forces 8D constraints.
        let levels: Vec<(&str, Vec<LatticeVector>)> = vec![
            ("Lambda_4096", enumerate_lambda_4096()),
            (
                "Lambda_2048",
                enumerate_lattice_by_predicate(is_in_lambda_2048),
            ),
            (
                "Lambda_1024",
                enumerate_lattice_by_predicate(is_in_lambda_1024),
            ),
            (
                "Lambda_512",
                enumerate_lattice_by_predicate(is_in_lambda_512),
            ),
            ("Lambda_256", enumerate_lambda_256()),
        ];

        for (name, vecs) in &levels {
            let (n, n_tri, n_sum, n_wt, n_l0, all_pass) = verify_octonion_parity_constraints(vecs);
            eprintln!("{name}: {n} vectors, all_parity_pass={all_pass}");
            assert!(
                all_pass,
                "{name}: parity violation! tri={n_tri}/{n}, sum={n_sum}/{n}, \
                 wt={n_wt}/{n}, l0={n_l0}/{n}"
            );
        }
    }
}
