//! Sparse vector representation for high-dimensional algebra states.
//!
//! # When to use this instead of a dense array
//!
//! At 1024D (DekaVoudon), a dense `[f64; 1024]` costs 8 KB per element.
//! At 2048D (Endekavoudon) it is 16 KB.  For physical states where most
//! components are zero -- QEC syndrome vectors, lattice-site wavefunctions,
//! sparse Fourier spectra, zero-divisor certificates -- sorted `(axis, value)`
//! pairs reduce both memory and arithmetic work significantly.
//!
//! # Data structure choice: sorted Vec vs HashMap
//!
//! Components are stored as a sorted `Vec<(usize, f64)>` rather than a
//! `HashMap<usize, f64>`.  The tradeoff:
//! - Hash maps give O(1) random access but have a large constant (~64 bytes
//!   overhead per entry on 64-bit) and poor cache locality for sequential scan.
//! - Sorted vecs give O(log k) random access but enable the **merge-join** inner
//!   product (`dot`): a single linear scan through both sorted lists in O(k1 + k2)
//!   that never touches either list more than once and fits in L1/L2 cache for
//!   physically reasonable k values (k < 1000).
//!
//! For the CD algebra use case the inner product is the dominant operation
//! (overlap integrals, violation rate estimation), so sorted Vec wins.
//!
//! # Type aliases
//!
//! `algebra_experimental::higher_cd::SparseApeironState` is a backward-compat
//! re-export of this type (added in Phase 4 of the CD tower modularisation).
//! `gororoba_algebra::construction::cd_tower::{Endekavoudon, Dodekvoudon, Dekatrisvoudon}`
//! are type aliases to this type for 2048D, 4096D, and 8192D respectively.

/// Sparse vector for a high-dimensional Cayley-Dickson algebra element or
/// any space whose dimension is indexed by a `usize`.
#[derive(Debug, Clone)]
pub struct SparseState {
    /// Total algebra dimension (must be power of 2 for CD use).
    pub dim: usize,
    /// Nonzero components as (axis_index, value), sorted by axis_index.
    pub components: Vec<(usize, f64)>,
}

impl SparseState {
    /// Create a zero state in the given dimension.
    pub fn zero(dim: usize) -> Self {
        Self { dim, components: Vec::new() }
    }

    /// Create from a dense slice, dropping components with |v| <= threshold.
    pub fn from_dense(dim: usize, values: &[f64], threshold: f64) -> Self {
        let components: Vec<(usize, f64)> = values
            .iter()
            .enumerate()
            .filter(|(_, v)| v.abs() > threshold)
            .map(|(i, &v)| (i, v))
            .collect();
        Self { dim, components }
    }

    /// Create from explicit (axis, value) pairs. Sorts by axis index.
    pub fn from_pairs(dim: usize, mut pairs: Vec<(usize, f64)>) -> Self {
        pairs.sort_by_key(|(idx, _)| *idx);
        Self { dim, components: pairs }
    }

    /// Number of nonzero components.
    pub fn nnz(&self) -> usize { self.components.len() }

    /// Sparsity ratio: 1.0 = fully zero, 0.0 = fully dense.
    pub fn sparsity(&self) -> f64 {
        if self.dim == 0 { return 1.0; }
        1.0 - (self.components.len() as f64 / self.dim as f64)
    }

    /// Squared L2 norm.
    pub fn norm_sq(&self) -> f64 {
        self.components.iter().map(|(_, v)| v * v).sum()
    }

    /// L2 norm.
    pub fn norm(&self) -> f64 { self.norm_sq().sqrt() }

    /// Inner product (overlap integral) via merge-join on sorted axis lists.
    ///
    /// # Algorithm
    ///
    /// Both `components` vectors are sorted by axis index.  We advance two
    /// cursors `i` and `j` through the two lists simultaneously:
    /// - If `axis_i < axis_j`: axis `i` has no counterpart in `other`; skip.
    /// - If `axis_i > axis_j`: axis `j` has no counterpart in `self`; skip.
    /// - If equal: accumulate `value_i * value_j` and advance both cursors.
    ///
    /// This is O(k1 + k2) in total comparisons with no allocation and at most
    /// one cache miss per unique axis encountered.  For k << dim (the typical
    /// case for physical CD states) this is orders of magnitude faster than
    /// densifying both vectors first.
    ///
    /// # Panics
    ///
    /// Does not check that `self.dim == other.dim`.  Call sites are responsible
    /// for ensuring dimension compatibility.
    pub fn dot(&self, other: &Self) -> f64 {
        let mut result = 0.0;
        let mut i = 0;
        let mut j = 0;
        while i < self.components.len() && j < other.components.len() {
            let (ai, vi) = self.components[i];
            let (aj, vj) = other.components[j];
            match ai.cmp(&aj) {
                std::cmp::Ordering::Less    => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal   => { result += vi * vj; i += 1; j += 1; }
            }
        }
        result
    }

    /// Convert to dense representation.
    pub fn to_dense(&self) -> Vec<f64> {
        let mut dense = vec![0.0; self.dim];
        for &(idx, val) in &self.components { dense[idx] = val; }
        dense
    }

    /// Scale all components in place.
    pub fn scale(&mut self, factor: f64) {
        for (_, v) in &mut self.components { *v *= factor; }
    }

    /// Add two sparse states, merging component lists.
    /// Near-zero sums (|sum| <= 1e-15) are dropped.
    pub fn add(&self, other: &Self) -> Self {
        let mut result = Vec::with_capacity(self.components.len() + other.components.len());
        let mut i = 0;
        let mut j = 0;
        while i < self.components.len() && j < other.components.len() {
            let (ai, vi) = self.components[i];
            let (aj, vj) = other.components[j];
            match ai.cmp(&aj) {
                std::cmp::Ordering::Less    => { result.push((ai, vi)); i += 1; }
                std::cmp::Ordering::Greater => { result.push((aj, vj)); j += 1; }
                std::cmp::Ordering::Equal   => {
                    let sum = vi + vj;
                    if sum.abs() > 1e-15 { result.push((ai, sum)); }
                    i += 1; j += 1;
                }
            }
        }
        while i < self.components.len() { result.push(self.components[i]); i += 1; }
        while j < other.components.len() { result.push(other.components[j]); j += 1; }
        Self { dim: self.dim, components: result }
    }

    /// Shannon entropy H = -sum(p_i ln p_i) of the magnitude distribution.
    /// H = 0 for a single-axis state; H = ln(k) for k equal components.
    pub fn shannon_entropy(&self) -> f64 {
        let total: f64 = self.components.iter().map(|(_, v)| v.abs()).sum();
        if total < 1e-15 { return 0.0; }
        let mut h = 0.0;
        for &(_, v) in &self.components {
            let p = v.abs() / total;
            if p > 1e-15 { h -= p * p.ln(); }
        }
        h
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_state() {
        let s = SparseState::zero(1024);
        assert_eq!(s.nnz(), 0);
        assert!((s.sparsity() - 1.0).abs() < 1e-10);
        assert!((s.norm() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn from_dense_threshold() {
        let mut dense = vec![0.0; 256];
        dense[0] = 1.0; dense[42] = -3.0; dense[100] = 0.5;
        let s = SparseState::from_dense(256, &dense, 1e-10);
        assert_eq!(s.nnz(), 3);
        assert!((s.norm_sq() - (1.0 + 9.0 + 0.25)).abs() < 1e-10);
    }

    #[test]
    fn dot_product() {
        let a = SparseState::from_pairs(128, vec![(0, 1.0), (5, 2.0), (10, 3.0)]);
        let b = SparseState::from_pairs(128, vec![(0, 4.0), (5, -1.0), (20, 7.0)]);
        assert!((a.dot(&b) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn add_with_cancellation() {
        let a = SparseState::from_pairs(64, vec![(0, 1.0), (5, 2.0)]);
        let b = SparseState::from_pairs(64, vec![(5, -2.0), (10, 3.0)]);
        let c = a.add(&b);
        assert_eq!(c.nnz(), 2);
        let dense = c.to_dense();
        assert!((dense[0] - 1.0).abs() < 1e-10);
        assert!((dense[5]).abs() < 1e-10);
        assert!((dense[10] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn shannon_entropy_single() {
        let s = SparseState::from_pairs(1024, vec![(42, 1.0)]);
        assert!((s.shannon_entropy() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn shannon_entropy_equal_pair() {
        let s = SparseState::from_pairs(1024, vec![(0, 1.0), (1, 1.0)]);
        assert!((s.shannon_entropy() - 2.0_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn roundtrip() {
        let pairs = vec![(3, 1.5), (17, -0.7), (255, 42.0)];
        let s = SparseState::from_pairs(256, pairs);
        let dense = s.to_dense();
        let s2 = SparseState::from_dense(256, &dense, 1e-10);
        assert_eq!(s.nnz(), s2.nnz());
        assert!((s.dot(&s2) - s.norm_sq()).abs() < 1e-10);
    }
}
