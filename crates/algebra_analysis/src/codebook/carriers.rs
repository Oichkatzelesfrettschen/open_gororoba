//! Layer 0: Typed carriers and carrier sets.
//!
//! The monograph abstraction hierarchy:
//!   * **Layer 0: TypedCarrier** -- a pair (basis_index, lattice_vec)
//!     pinning a CD basis element to its lattice-encoded form
//!   * **Layer 1: EncodingDictionary** -- a validated bijection
//!     basis <-> lattice (defined in the parent module)
//!   * **Layer 2: Elevated addition** -- (in `elevated_addition`)
//!   * **Layer 3+: Named graph predicates, invariants** -- elsewhere
//!
//! This submodule provides Layer 0 plus the `CarrierSet` builder/
//! validator that Layer 1 consumes:
//!   * `TypedCarrier`           -- single carrier pair
//!   * `CarrierSet`             -- collection with bijection invariants
//!   * `CarrierSetValidation`   -- validation report
//!
//! All three are pub at the submodule level and re-exported from the
//! parent module via `pub use` so external paths
//! `algebra_analysis::codebook::{TypedCarrier, CarrierSet,
//! CarrierSetValidation}` remain stable.

use std::collections::HashMap;

use super::lambda_predicates::{
    LatticeVector, is_in_base_universe, is_in_lambda_256, is_in_lambda_512, is_in_lambda_1024,
    is_in_lambda_2048,
};

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
        Self {
            basis_index,
            lattice_vec,
        }
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
        Some(Self {
            basis_index,
            lattice_vec: lv,
        })
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
        let mut carriers: Vec<TypedCarrier> = map
            .iter()
            .filter_map(|(&idx, v)| TypedCarrier::from_i32_vec(idx, v))
            .collect();
        carriers.sort_by_key(|c| c.basis_index);

        let index: HashMap<usize, usize> = carriers
            .iter()
            .enumerate()
            .map(|(pos, c)| (c.basis_index, pos))
            .collect();

        Self {
            dim,
            carriers,
            index,
        }
    }

    /// Build a carrier set from pre-validated LatticeVectors.
    pub fn from_lattice_vecs(dim: usize, pairs: &[(usize, LatticeVector)]) -> Self {
        let mut carriers: Vec<TypedCarrier> = pairs
            .iter()
            .map(|&(idx, lv)| TypedCarrier::new(idx, lv))
            .collect();
        carriers.sort_by_key(|c| c.basis_index);

        let index: HashMap<usize, usize> = carriers
            .iter()
            .enumerate()
            .map(|(pos, c)| (c.basis_index, pos))
            .collect();

        Self {
            dim,
            carriers,
            index,
        }
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
        self.carriers
            .iter()
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
