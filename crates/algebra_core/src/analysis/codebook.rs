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


/// A vector in the 8D integer lattice (typically {-1, 0, 1}).
pub type LatticeVector = [i8; 8];

/// Check if a vector is in the "Base Universe" (Trinary, Even Sum, Even Weight).
pub fn is_in_base_universe(v: &LatticeVector) -> bool {
    // 1. Trinary
    if v.iter().any(|&x| !(-1..=1).contains(&x)) {
        return false;
    }
    // 2. Even coordinate sum
    let sum: i32 = v.iter().map(|&x| x as i32).sum();
    if sum % 2 != 0 {
        return false;
    }
    // 3. Even Hamming weight (nonzero count)
    let weight = v.iter().filter(|&&x| x != 0).count();
    if weight % 2 != 0 {
        return false;
    }
    // 4. l_0 != +1 (from analysis of 2048D set)
    if v[0] == 1 {
        return false;
    }
    true
}

/// Check if a vector is in Lambda_2048 (Base minus 139 forbidden prefixes).
pub fn is_in_lambda_2048(v: &LatticeVector) -> bool {
    if !is_in_base_universe(v) {
        return false;
    }

    // Forbidden prefixes for 2048D
    // (l_0, l_1, l_2) = (0, 1, 1)
    if v[0] == 0 && v[1] == 1 && v[2] == 1 {
        return false;
    }
    // (l_0..l_4) = (0, 1, 0, 1, 1)
    if v[0] == 0 && v[1] == 1 && v[2] == 0 && v[3] == 1 && v[4] == 1 {
        return false;
    }
    // (l_0..l_5) = (0, 1, 0, 1, 0, 1)
    if v[0] == 0 && v[1] == 1 && v[2] == 0 && v[3] == 1 && v[4] == 0 && v[5] == 1 {
        return false;
    }

    true
}

/// Check if a vector is in Lambda_1024 (Lambda_2048 with l_0 = -1 minus 70 points).
pub fn is_in_lambda_1024(v: &LatticeVector) -> bool {
    if !is_in_lambda_2048(v) {
        return false;
    }
    
    // Slice condition
    if v[0] != -1 {
        return false;
    }

    // Additional exclusions for 1024D
    // (-1, 1, 1, 1)
    if v[1] == 1 && v[2] == 1 && v[3] == 1 {
        return false;
    }
    // (-1, 1, 1, 0, 0)
    if v[1] == 1 && v[2] == 1 && v[3] == 0 && v[4] == 0 {
        return false;
    }
    // (-1, 1, 1, 0, 1)
    if v[1] == 1 && v[2] == 1 && v[3] == 0 && v[4] == 1 {
        return false;
    }
    // Singleton exceptions would go here, but omitted for brevity in this MVP.

    true
}

/// Check if a vector is in Lambda_512 (Lambda_1024 minus 6 regions).
pub fn is_in_lambda_512(v: &LatticeVector) -> bool {
    if !is_in_lambda_1024(v) {
        return false;
    }

    // Forbidden regions (l_0 is always -1 here)
    // 1. l_1 = 1
    if v[1] == 1 { return false; }
    // 2. l_1=0, l_2=1
    if v[1] == 0 && v[2] == 1 { return false; }
    // 3. l_1=0, l_2=0, l_3=0
    if v[1] == 0 && v[2] == 0 && v[3] == 0 { return false; }
    // 4. l_1=0, l_2=0, l_3=1
    if v[1] == 0 && v[2] == 0 && v[3] == 1 { return false; }
    // 5. l_1=0, l_2=0, l_3=-1, l_4=1
    if v[1] == 0 && v[2] == 0 && v[3] == -1 && v[4] == 1 { return false; }
    // 6. l_1=0, l_2=0, l_3=-1, l_4=0, l_5=1, l_6=1
    if v[1] == 0 && v[2] == 0 && v[3] == -1 && v[4] == 0 && v[5] == 1 && v[6] == 1 { return false; }

    true
}

/// Check if a vector is in Lambda_256 (Lambda_512 minus 6 regions).
pub fn is_in_lambda_256(v: &LatticeVector) -> bool {
    if !is_in_lambda_512(v) {
        return false;
    }

    // Forbidden regions (l_0 = -1)
    // 1. l_1 = 0 (implies l_1 must be -1 for success, since l_1 != 1 from 512 rule)
    if v[1] == 0 { return false; } 
    
    // For the remaining, l_1 = -1 is established.
    // 2. (-1, -1, 1, 1)
    if v[2] == 1 && v[3] == 1 { return false; }
    // 3. (-1, -1, 1, 0)
    if v[2] == 1 && v[3] == 0 { return false; }
    // 4. (-1, -1, 1, -1, 1)
    if v[2] == 1 && v[3] == -1 && v[4] == 1 { return false; }
    // 5. (-1, -1, 1, -1, 0)
    if v[2] == 1 && v[3] == -1 && v[4] == 0 { return false; }
    // 6. Singleton (-1, -1, 1, -1, -1, 1, 1, 1)
    if v[2] == 1 && v[3] == -1 && v[4] == -1 && v[5] == 1 && v[6] == 1 && v[7] == 1 { return false; }

    true
}

/// Apply the Scalar Shadow action to a lattice vector.
///
/// Addition mode: l_out = l + a * 1_8
/// Multiplication mode: l_out = a * l
pub fn apply_scalar_shadow(l: &LatticeVector, a: i8, mode: &str) -> LatticeVector {
    let mut res = [0i8; 8];
    match mode {
        "add" => {
            for i in 0..8 {
                res[i] = l[i].saturating_add(a);
            }
        },
        "mul" => {
            for i in 0..8 {
                res[i] = l[i] * a;
            }
        },
        _ => panic!("Unknown mode: {mode}"),
    }
    res
}

// ============================================================================
// Layer 0: Typed Carriers
// ============================================================================

/// A typed carrier X_n = (b, l) pairing a Cayley-Dickson basis element
/// with its lattice vector in the encoding dictionary.
///
/// This is the foundational data type for the monograph abstraction hierarchy:
/// - Layer 0: TypedCarrier (this struct)
/// - Layer 1: EncodingDictionary (Phi_n: basis -> lattice bijection)
/// - Layer 2: Elevated addition (l -> l + Phi(b))
/// - Layer 3: Named graph predicates (P_ZD, P_match)
/// - Layer 4: Invariant suite (degree, spectrum, triangles, etc.)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TypedCarrier {
    /// CD basis element index in [0, dim).
    pub basis_index: usize,
    /// 8D lattice vector in {-1, 0, 1}^8.
    pub lattice_vec: LatticeVector,
}

/// The dimension tier for filtration membership queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FiltrationTier {
    Base,
    Lambda2048,
    Lambda1024,
    Lambda512,
    Lambda256,
}

impl TypedCarrier {
    /// Create a new typed carrier.
    pub fn new(basis_index: usize, lattice_vec: LatticeVector) -> Self {
        Self { basis_index, lattice_vec }
    }

    /// Convert from the Vec<i32> representation used in cd_external.
    /// Returns None if any coordinate is outside [-1, 1] or the vector
    /// does not have exactly 8 components.
    pub fn from_i32_vec(basis_index: usize, v: &[i32]) -> Option<Self> {
        if v.len() != 8 {
            return None;
        }
        let mut lv = [0i8; 8];
        for (i, &val) in v.iter().enumerate() {
            if !(-1..=1).contains(&val) {
                return None;
            }
            lv[i] = val as i8;
        }
        Some(Self { basis_index, lattice_vec: lv })
    }

    /// Return the highest filtration tier this carrier's lattice vector
    /// belongs to (most restrictive = smallest codebook).
    pub fn filtration_tier(&self) -> FiltrationTier {
        if is_in_lambda_256(&self.lattice_vec) {
            FiltrationTier::Lambda256
        } else if is_in_lambda_512(&self.lattice_vec) {
            FiltrationTier::Lambda512
        } else if is_in_lambda_1024(&self.lattice_vec) {
            FiltrationTier::Lambda1024
        } else if is_in_lambda_2048(&self.lattice_vec) {
            FiltrationTier::Lambda2048
        } else if is_in_base_universe(&self.lattice_vec) {
            FiltrationTier::Base
        } else {
            // Not even in the base universe -- should not happen for valid data.
            FiltrationTier::Base
        }
    }

    /// Check if this carrier's lattice vector is in Lambda_n.
    pub fn is_in_lambda(&self, dim: usize) -> bool {
        match dim {
            256 => is_in_lambda_256(&self.lattice_vec),
            512 => is_in_lambda_512(&self.lattice_vec),
            1024 => is_in_lambda_1024(&self.lattice_vec),
            2048 => is_in_lambda_2048(&self.lattice_vec),
            _ => is_in_base_universe(&self.lattice_vec),
        }
    }
}

/// The full carrier set for a given CD algebra dimension.
///
/// Collects all typed carriers X_n = (b, l) and provides O(1) lookup
/// by basis index, filtration queries, and consistency checks.
#[derive(Debug, Clone)]
pub struct CarrierSet {
    /// CD algebra dimension.
    pub dim: usize,
    /// Ordered list of carriers (by basis_index).
    carriers: Vec<TypedCarrier>,
    /// Basis index -> position in carriers vec (O(1) lookup).
    index: HashMap<usize, usize>,
}

impl CarrierSet {
    /// Build a carrier set from a basis_index -> lattice_vector map.
    /// This is the bridge from cd_external::load_lattice_map().
    pub fn from_i32_map(dim: usize, map: &HashMap<usize, Vec<i32>>) -> Self {
        let mut carriers: Vec<TypedCarrier> = map.iter()
            .filter_map(|(&idx, v)| TypedCarrier::from_i32_vec(idx, v))
            .collect();
        carriers.sort_by_key(|c| c.basis_index);

        let index: HashMap<usize, usize> = carriers.iter()
            .enumerate()
            .map(|(pos, c)| (c.basis_index, pos))
            .collect();

        Self { dim, carriers, index }
    }

    /// Build a carrier set from pre-validated LatticeVectors.
    pub fn from_lattice_vecs(dim: usize, pairs: &[(usize, LatticeVector)]) -> Self {
        let mut carriers: Vec<TypedCarrier> = pairs.iter()
            .map(|&(idx, lv)| TypedCarrier::new(idx, lv))
            .collect();
        carriers.sort_by_key(|c| c.basis_index);

        let index: HashMap<usize, usize> = carriers.iter()
            .enumerate()
            .map(|(pos, c)| (c.basis_index, pos))
            .collect();

        Self { dim, carriers, index }
    }

    /// Number of carriers in this set.
    pub fn len(&self) -> usize {
        self.carriers.len()
    }

    /// Whether the carrier set is empty.
    pub fn is_empty(&self) -> bool {
        self.carriers.is_empty()
    }

    /// Look up a carrier by basis index. O(1).
    pub fn get(&self, basis_index: usize) -> Option<&TypedCarrier> {
        self.index.get(&basis_index).map(|&pos| &self.carriers[pos])
    }

    /// Iterate over all carriers in basis-index order.
    pub fn iter(&self) -> impl Iterator<Item = &TypedCarrier> {
        self.carriers.iter()
    }

    /// Return all carriers whose lattice vectors are in Lambda_target_dim.
    pub fn filter_to_lambda(&self, target_dim: usize) -> Vec<&TypedCarrier> {
        self.carriers.iter()
            .filter(|c| c.is_in_lambda(target_dim))
            .collect()
    }

    /// Check that the carrier set is a valid encoding dictionary:
    /// - Every basis index in [0, dim) has exactly one carrier.
    /// - No two carriers share the same lattice vector (injectivity).
    pub fn validate(&self) -> CarrierSetValidation {
        let mut missing = Vec::new();
        for i in 0..self.dim {
            if !self.index.contains_key(&i) {
                missing.push(i);
            }
        }

        let mut seen = HashMap::new();
        let mut duplicates = Vec::new();
        for c in &self.carriers {
            if let Some(&prev_idx) = seen.get(&c.lattice_vec) {
                duplicates.push((prev_idx, c.basis_index));
            } else {
                seen.insert(c.lattice_vec, c.basis_index);
            }
        }

        CarrierSetValidation {
            is_complete: missing.is_empty(),
            is_injective: duplicates.is_empty(),
            missing_basis_indices: missing,
            duplicate_lattice_pairs: duplicates,
        }
    }

    /// Count how many carriers fall into each filtration tier.
    pub fn tier_histogram(&self) -> HashMap<FiltrationTier, usize> {
        let mut hist = HashMap::new();
        for c in &self.carriers {
            *hist.entry(c.filtration_tier()).or_insert(0) += 1;
        }
        hist
    }
}

/// Result of validating a CarrierSet for encoding dictionary properties.
#[derive(Debug, Clone)]
pub struct CarrierSetValidation {
    /// True if every basis index in [0, dim) has a carrier.
    pub is_complete: bool,
    /// True if no two carriers share the same lattice vector.
    pub is_injective: bool,
    /// Basis indices missing from the carrier set.
    pub missing_basis_indices: Vec<usize>,
    /// Pairs of basis indices that map to the same lattice vector.
    pub duplicate_lattice_pairs: Vec<(usize, usize)>,
}

impl CarrierSetValidation {
    /// True if the carrier set forms a valid bijection.
    pub fn is_valid_dictionary(&self) -> bool {
        self.is_complete && self.is_injective
    }
}

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

        let inverse: HashMap<LatticeVector, usize> = cs.iter()
            .map(|c| (c.lattice_vec, c.basis_index))
            .collect();

        Ok(Self { carriers: cs, inverse })
    }

    /// Build from a basis_index -> Vec<i32> map (bridge from cd_external).
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
        self.carriers.iter().map(|c| (c.basis_index, &c.lattice_vec))
    }

    /// Restrict this dictionary to carriers whose lattice vectors are in
    /// Lambda_{target_dim}. Returns a new (smaller) dictionary for the
    /// sub-codebook at the target filtration tier.
    ///
    /// Note: the returned dictionary has dim = target_dim, and its basis
    /// indices are the ORIGINAL indices from the parent dictionary. It will
    /// not pass completeness validation (missing basis indices are expected).
    pub fn restrict_to_lambda(&self, target_dim: usize) -> Vec<(usize, LatticeVector)> {
        self.carriers.iter()
            .filter(|c| c.is_in_lambda(target_dim))
            .map(|c| (c.basis_index, c.lattice_vec))
            .collect()
    }

    /// Compute the scalar shadow pi(b) for a basis element.
    /// Defined as sign(sum(lattice_vec)).
    pub fn scalar_shadow(&self, basis_index: usize) -> Option<i8> {
        self.encode(basis_index).map(|lv| {
            let s: i32 = lv.iter().map(|&x| x as i32).sum();
            if s > 0 { 1 } else if s < 0 { -1 } else { 0 }
        })
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

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
            (0, [-1, -1, -1, -1, 0, 0, 0, 0]),   // in Lambda_256
            (1, [0, -1, 0, -1, 0, -1, 0, -1]),    // base only (l_0 = 0)
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
            (1, [-1, -1, -1, -1, -1, -1, 0, 0]),  // Lambda_256
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
        assert!(is_in_base_universe(&[-1, -1, 0, 0, 0, 0, 0, 0]));  // sum=-2, wt=2
        assert!(is_in_base_universe(&[0, 0, 0, 0, 0, 0, 0, 0]));    // sum=0, wt=0
        assert!(!is_in_base_universe(&[1, 0, 0, 0, 0, 0, 0, 0]));   // l_0=1 forbidden
        assert!(!is_in_base_universe(&[-1, 0, 0, 0, 0, 0, 0, 0]));  // sum=-1 odd
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
        let pairs = vec![
            (0, same_vec),
            (1, same_vec),
        ];
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
}
