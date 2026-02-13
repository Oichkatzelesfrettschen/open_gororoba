//! Survival depth spectrum for sedenion multiplication pairs.
//!
//! Maps each (i, j) basis pair to a "survival depth" measuring how deeply
//! the product cd_basis_mul(i, j) survives in the Patricia trie prefix
//! hierarchy. Deeper survival implies greater structural stability under
//! filtration, which we interpret as higher effective mass.

use crate::basis_index::BasisIndexCodec;
use crate::patricia_trie::PatriciaIndex;
use algebra_core::construction::cayley_dickson::cd_basis_mul_sign;
use std::collections::HashMap;

/// Dimension of sedenion algebra.
const SEDENION_DIM: usize = 16;

/// Survival depth for a single basis multiplication pair.
#[derive(Debug, Clone, Copy)]
pub struct SurvivalEntry {
    /// Left basis index (0..15)
    pub lhs: usize,
    /// Right basis index (0..15)
    pub rhs: usize,
    /// Product basis index (lhs XOR rhs)
    pub product_basis: usize,
    /// Product sign from CD multiplication
    pub product_sign: i32,
    /// Depth in Patricia trie: shared prefix bits with the zero-divisor anchor
    pub depth: u32,
}

/// Maps sedenion pairs to survival depths via Patricia trie analysis.
///
/// The survival depth measures structural persistence: pairs whose products
/// share long bit prefixes with the zero-divisor anchor (basis=0, sign=-1)
/// are "closer to annihilation" and survive less deeply. Pairs far from
/// the anchor survive deeply.
#[derive(Debug, Clone)]
pub struct SurvivalDepthMap {
    /// All 256 entries (16 x 16 basis pairs)
    pub entries: Vec<SurvivalEntry>,
    /// Quick lookup: (lhs, rhs) -> depth
    depth_map: HashMap<(u8, u8), u32>,
}

impl SurvivalDepthMap {
    /// Compute survival depths for all sedenion basis pairs.
    ///
    /// Workflow:
    /// 1. Encode each product (basis, sign) via BasisIndexCodec
    /// 2. Insert all encoded products into PatriciaIndex
    /// 3. Measure shared prefix bits with zero-divisor anchor
    pub fn compute() -> Self {
        let codec = BasisIndexCodec::new(SEDENION_DIM);
        let mut trie = PatriciaIndex::new();
        let zd_anchor = codec.encode(0, -1);

        let mut entries = Vec::with_capacity(SEDENION_DIM * SEDENION_DIM);
        let mut depth_map = HashMap::with_capacity(SEDENION_DIM * SEDENION_DIM);

        for i in 0..SEDENION_DIM {
            for j in 0..SEDENION_DIM {
                let sign = cd_basis_mul_sign(SEDENION_DIM, i, j);
                let basis = i ^ j;
                let key = codec.encode(basis, sign);
                trie.insert(key);

                let depth = PatriciaIndex::shared_prefix_bits(zd_anchor, key);

                entries.push(SurvivalEntry {
                    lhs: i,
                    rhs: j,
                    product_basis: basis,
                    product_sign: sign,
                    depth,
                });
                depth_map.insert((i as u8, j as u8), depth);
            }
        }

        Self { entries, depth_map }
    }

    /// Look up survival depth for a specific pair.
    pub fn depth(&self, lhs: usize, rhs: usize) -> u32 {
        self.depth_map
            .get(&(lhs as u8, rhs as u8))
            .copied()
            .unwrap_or(0)
    }

    /// Number of distinct depth values observed.
    pub fn n_distinct_depths(&self) -> usize {
        let depths: std::collections::HashSet<u32> = self.entries.iter().map(|e| e.depth).collect();
        depths.len()
    }

    /// Return the minimum and maximum depths.
    pub fn depth_range(&self) -> (u32, u32) {
        let min = self.entries.iter().map(|e| e.depth).min().unwrap_or(0);
        let max = self.entries.iter().map(|e| e.depth).max().unwrap_or(0);
        (min, max)
    }

    /// Mean survival depth across all 256 pairs.
    pub fn mean_depth(&self) -> f64 {
        if self.entries.is_empty() {
            return 0.0;
        }
        let sum: u64 = self.entries.iter().map(|e| e.depth as u64).sum();
        sum as f64 / self.entries.len() as f64
    }
}

/// Histogram bin for survival depth distribution.
#[derive(Debug, Clone, Copy)]
pub struct DepthHistogramBin {
    pub depth: u32,
    pub count: usize,
    pub fraction: f64,
}

/// Compute histogram of survival depths across all 256 basis pairs.
pub fn depth_histogram(map: &SurvivalDepthMap) -> Vec<DepthHistogramBin> {
    let mut counts: HashMap<u32, usize> = HashMap::new();
    for entry in &map.entries {
        *counts.entry(entry.depth).or_insert(0) += 1;
    }

    let total = map.entries.len() as f64;
    let mut bins: Vec<DepthHistogramBin> = counts
        .into_iter()
        .map(|(depth, count)| DepthHistogramBin {
            depth,
            count,
            fraction: count as f64 / total,
        })
        .collect();

    bins.sort_by_key(|b| b.depth);
    bins
}

/// Cluster entries by depth, returning groups sorted by depth.
///
/// Each cluster is a depth level with all pairs at that depth.
/// Clusters with the most entries are candidates for mass peaks.
pub fn depth_clusters(map: &SurvivalDepthMap) -> Vec<(u32, Vec<SurvivalEntry>)> {
    let mut groups: HashMap<u32, Vec<SurvivalEntry>> = HashMap::new();
    for entry in &map.entries {
        groups.entry(entry.depth).or_default().push(*entry);
    }

    let mut sorted: Vec<(u32, Vec<SurvivalEntry>)> = groups.into_iter().collect();
    sorted.sort_by_key(|(depth, _)| *depth);
    sorted
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_survival_depth_map_has_256_entries() {
        let map = SurvivalDepthMap::compute();
        assert_eq!(map.entries.len(), 256);
    }

    #[test]
    fn test_depth_map_lookup_consistent() {
        let map = SurvivalDepthMap::compute();
        for entry in &map.entries {
            assert_eq!(map.depth(entry.lhs, entry.rhs), entry.depth);
        }
    }

    #[test]
    fn test_identity_product_depth() {
        let map = SurvivalDepthMap::compute();
        // e_0 * e_0 = e_0 (identity), product=(0, +1)
        // Zero-div anchor is (0, -1). Encoded: (0<<1)|1 = 1 vs (0<<1)|0 = 0
        // XOR = 1, leading zeros = 63, so depth = 63
        let d = map.depth(0, 0);
        assert!(d > 0, "Identity product should have nonzero depth");
    }

    #[test]
    fn test_depth_range_valid() {
        let map = SurvivalDepthMap::compute();
        let (min, max) = map.depth_range();
        assert!(min <= max);
        assert!(max <= 64);
    }

    #[test]
    fn test_mean_depth_reasonable() {
        let map = SurvivalDepthMap::compute();
        let mean = map.mean_depth();
        assert!(mean > 0.0, "Mean depth should be positive");
        assert!(mean <= 64.0, "Mean depth cannot exceed 64 bits");
    }

    #[test]
    fn test_distinct_depths_at_least_two() {
        let map = SurvivalDepthMap::compute();
        assert!(
            map.n_distinct_depths() >= 2,
            "Should have at least 2 distinct depths"
        );
    }

    #[test]
    fn test_histogram_sums_to_256() {
        let map = SurvivalDepthMap::compute();
        let hist = depth_histogram(&map);
        let total: usize = hist.iter().map(|b| b.count).sum();
        assert_eq!(total, 256);
    }

    #[test]
    fn test_histogram_fractions_sum_to_one() {
        let map = SurvivalDepthMap::compute();
        let hist = depth_histogram(&map);
        let frac_sum: f64 = hist.iter().map(|b| b.fraction).sum();
        assert!(
            (frac_sum - 1.0).abs() < 1e-12,
            "Fractions should sum to 1.0, got {}",
            frac_sum
        );
    }

    #[test]
    fn test_histogram_sorted_by_depth() {
        let map = SurvivalDepthMap::compute();
        let hist = depth_histogram(&map);
        for w in hist.windows(2) {
            assert!(w[0].depth <= w[1].depth);
        }
    }

    #[test]
    fn test_depth_clusters_cover_all_entries() {
        let map = SurvivalDepthMap::compute();
        let clusters = depth_clusters(&map);
        let total: usize = clusters.iter().map(|(_, entries)| entries.len()).sum();
        assert_eq!(total, 256);
    }

    #[test]
    fn test_product_basis_is_xor() {
        let map = SurvivalDepthMap::compute();
        for entry in &map.entries {
            assert_eq!(entry.product_basis, entry.lhs ^ entry.rhs);
        }
    }
}
