//! Lattice-filtration verification cluster (Thesis B/C + Phase 1.4).
//!
//! Verifies that the four canonical CD codebooks form a strict
//! nesting `Lambda_256 < Lambda_512 < Lambda_1024 < Lambda_2048`,
//! and characterizes each nesting step as a lexicographic prefix
//! cut. Also exposes `enumerate_base_universe`, the base-universe
//! S_base used by downstream filtration analyses.
//!
//! Public surface:
//!   * `FiltrationResult`        -- nesting verification record
//!   * `LexPrefixCut`            -- single-step prefix-cut rule
//!   * `verify_lattice_filtration`
//!   * `learn_prefix_cut`
//!   * `verify_prefix_cut`
//!   * `learn_full_filtration_cuts`
//!   * `enumerate_base_universe`
//!
//! Re-exported from `cd_external` via `pub use` so external paths
//! algebra_experimental::cd_external::{FiltrationResult, ...}
//! remain stable.

use std::collections::BTreeSet;

use super::load_lattice_points;

// ---------------------------------------------------------------------------

/// Result of lattice filtration nesting verification.
#[derive(Debug)]
pub struct FiltrationResult {
    /// Sizes of each codebook level, from smallest to largest.
    pub sizes: Vec<(usize, usize)>,
    /// Whether each level is a strict subset of the next.
    pub strict_subsets: Vec<bool>,
    /// Whether the full filtration forms a strict chain.
    pub is_strict_chain: bool,
}

/// Verify that lattice codebooks form a strict filtration:
/// Lambda_256 < Lambda_512 < Lambda_1024 < Lambda_2048.
pub fn verify_lattice_filtration() -> FiltrationResult {
    let dims = [256, 512, 1024, 2048];
    let sets: Vec<BTreeSet<Vec<i32>>> = dims
        .iter()
        .map(|&d| load_lattice_points(d).into_iter().collect())
        .collect();

    let sizes: Vec<(usize, usize)> = dims
        .iter()
        .zip(sets.iter())
        .map(|(&d, s)| (d, s.len()))
        .collect();

    let strict_subsets: Vec<bool> = (0..sets.len() - 1)
        .map(|i| sets[i].is_subset(&sets[i + 1]) && sets[i].len() < sets[i + 1].len())
        .collect();

    let is_strict_chain = strict_subsets.iter().all(|&b| b);

    FiltrationResult {
        sizes,
        strict_subsets,
        is_strict_chain,
    }
}

// ---------------------------------------------------------------------------
// Thesis C: Prefix-Cut Characterization
// ---------------------------------------------------------------------------

/// A lexicographic prefix-cut rule: the child codebook is exactly the set of
/// parent points that are lexicographically <= the boundary point.
#[derive(Debug, Clone)]
pub struct LexPrefixCut {
    /// The boundary point (last included point in lex order).
    pub boundary: Vec<i32>,
    /// Number of points in the parent codebook.
    pub parent_size: usize,
    /// Number of points in the child codebook.
    pub child_size: usize,
    /// Index in the 8D coordinate vector where the cut diverges.
    pub divergence_coord: usize,
}

/// Learn prefix-cut rules between adjacent filtration levels.
///
/// Returns the lexicographic boundary point that exactly separates the child
/// codebook from its complement within the parent. This works because our
/// lattice codebooks are nested by lexicographic prefix: the child is exactly
/// the first N points of the parent in lexicographic order.
pub fn learn_prefix_cut(
    parent: &BTreeSet<Vec<i32>>,
    child: &BTreeSet<Vec<i32>>,
) -> Option<LexPrefixCut> {
    if !child.is_subset(parent) || child.len() >= parent.len() {
        return None;
    }

    let parent_sorted: Vec<&Vec<i32>> = parent.iter().collect();
    let n_child = child.len();

    // Verify: first n_child elements of parent (lex order) == child
    let lex_first: BTreeSet<Vec<i32>> = parent_sorted[..n_child]
        .iter()
        .map(|&v| v.clone())
        .collect();

    if lex_first != *child {
        return None; // Not a lexicographic prefix cut
    }

    let boundary = parent_sorted[n_child - 1].clone();
    let first_excluded = parent_sorted[n_child];

    // Find divergence coordinate
    let divergence_coord = boundary
        .iter()
        .zip(first_excluded.iter())
        .position(|(a, b)| a != b)
        .unwrap_or(7);

    Some(LexPrefixCut {
        boundary,
        parent_size: parent.len(),
        child_size: child.len(),
        divergence_coord,
    })
}

/// Verify that a prefix-cut rule exactly partitions the parent into child + excluded.
pub fn verify_prefix_cut(
    parent: &BTreeSet<Vec<i32>>,
    child: &BTreeSet<Vec<i32>>,
    cut: &LexPrefixCut,
) -> bool {
    let included: BTreeSet<Vec<i32>> = parent
        .iter()
        .filter(|p| p.as_slice() <= cut.boundary.as_slice())
        .cloned()
        .collect();
    included == *child
}

/// Learn all prefix-cut rules for the full filtration chain.
pub fn learn_full_filtration_cuts() -> Vec<(usize, usize, LexPrefixCut)> {
    let dims = [256, 512, 1024, 2048];
    let sets: Vec<BTreeSet<Vec<i32>>> = dims
        .iter()
        .map(|&d| load_lattice_points(d).into_iter().collect())
        .collect();

    let mut cuts = Vec::new();
    for i in 0..sets.len() - 1 {
        if let Some(cut) = learn_prefix_cut(&sets[i + 1], &sets[i]) {
            cuts.push((dims[i + 1], dims[i], cut));
        }
    }
    cuts
}

// ---------------------------------------------------------------------------
// Base Universe and Exclusion (Phase 1.4)
// ---------------------------------------------------------------------------

/// Enumerate the base universe S_base: all vectors in {-1,0,1}^8 satisfying
/// `coord[0] != +1`, even sum, and even nonzero count.
pub fn enumerate_base_universe() -> BTreeSet<Vec<i32>> {
    let mut result = BTreeSet::new();
    // coord[0] in {-1, 0}, coords[1..8] in {-1, 0, 1}
    // Total: 2 * 3^7 = 4374 candidates before parity filter
    let vals: [i32; 3] = [-1, 0, 1];

    for &c0 in &[-1i32, 0] {
        for &c1 in &vals {
            for &c2 in &vals {
                for &c3 in &vals {
                    for &c4 in &vals {
                        for &c5 in &vals {
                            for &c6 in &vals {
                                for &c7 in &vals {
                                    let v = vec![c0, c1, c2, c3, c4, c5, c6, c7];
                                    let s: i32 = v.iter().sum();
                                    let nz: usize = v.iter().filter(|&&x| x != 0).count();
                                    if (s as usize).is_multiple_of(2) && nz.is_multiple_of(2) {
                                        result.insert(v);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    result
}

