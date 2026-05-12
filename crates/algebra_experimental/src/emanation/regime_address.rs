//! (s,g)-Modularity recursive regime address (L9b) and the hide/fill
//! row-degree-invariance analysis (L9c).
//!
//! The DMZ count of a strut constant S at CD level N is determined by a
//! recursive "regime address" -- a binary vector of length N-4.
//!
//! At each level k (from N down to 5), the "half-generator" g_k = 2^(k-2)
//! splits S into a lower band (S <= g_k) and upper band (S > g_k):
//!   - Lower band: inherit regime from level k-1
//!   - Upper band: new regime, sub-classified by regime(k-1, S - g_k)
//!   - Powers of 2 >= 8 are generators, always full-fill (merge with mandala)
//!
//! This produces exactly 2^(N-4) regime classes, with regime count doubling
//! at each CD level (de Marrais's "regime-doubling cascade").
//!
//! Hide/fill: within each regime (same regime address), all strut
//! constants produce ETs with the same sorted row-degree distribution.
//! Sky struts hide cells from certain rows; the hidden pattern permutes
//! across strut constants within the regime, while the union covers all
//! addressable cells (collective coverage).

use super::strutted_et::{StruttedEmanationTable, create_strutted_et};

// ===========================================================================
// L9b: (s,g)-Modularity -- Recursive Regime Address
// ===========================================================================

/// Compute the recursive regime address for strut constant `s` at CD level `n`.
///
/// Returns a binary vector of length `n - 4` (empty for sedenions).
/// Two struts with the same regime address always have the same DMZ count.
pub fn regime_address(n: usize, s: usize) -> Vec<u8> {
    if n <= 4 {
        return vec![];
    }
    let g = 1usize << (n - 2); // Half-generator = generator of level N-1

    // Generators (powers of 2 >= 8) always full-fill.
    // Map to S=3 (an unambiguous mandala value) to avoid recursion issues.
    if s >= 8 && s.is_power_of_two() {
        return regime_address(n, 3);
    }

    if s <= g {
        let mut addr = vec![0u8];
        addr.extend(regime_address(n - 1, s));
        addr
    } else {
        let remainder = s - g;
        let mut addr = vec![1u8];
        addr.extend(regime_address(n - 1, remainder));
        addr
    }
}

/// Number of distinct DMZ regimes at CD level `n`.
///
/// Returns 2^(n-4): sedenions have 1, pathions 2, chingons 4, routons 8.
pub fn regime_count(n: usize) -> usize {
    1usize << n.saturating_sub(4)
}

// ===========================================================================
// L9c: Hide/Fill Involution -- DMZ Row-Degree Invariance
// ===========================================================================
//
// Within each regime (same regime_address), all strut constants produce ETs
// with the same sorted row-degree distribution. This is a stronger invariant
// than just DMZ count: it constrains the *shape* of the fill pattern.
//
// Key properties verified:
//   1. Mandala regime is always "full fill" (every addressable cell is DMZ,
//      uniform row degree = K-2 where K = 2^(N-1) - 2).
//   2. Sky struts have non-uniform row degrees: some rows keep full fill,
//      others drop to a lower degree.
//   3. The row-degree distribution is a regime invariant: permuted by S
//      but identical when sorted.
//   4. Sky UNION covers all addressable cells (collective coverage).
//
// De Marrais calls this "hide/fill": mandala shows all; crossing into the
// sky band "hides" cells from certain rows; the hidden pattern permutes
// across strut constants within the regime.

/// The sorted row-degree distribution of a strutted ET.
///
/// For a K x K ET, this is a vector of length K where entry i is the
/// number of DMZ cells in the i-th row (after sorting ascending).
/// Two ETs with the same sorted row-degree distribution have the same
/// "fill shape" even if individual cell positions differ.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RowDegreeDistribution {
    /// Sorted ascending row degrees.
    pub degrees: Vec<usize>,
    /// Total DMZ count (sum of degrees).
    pub dmz_total: usize,
    /// Grid size K.
    pub k: usize,
}

/// Compute the sorted row-degree distribution of a strutted ET.
pub fn row_degree_distribution(et: &StruttedEmanationTable) -> RowDegreeDistribution {
    let k = et.tone_row.k;
    let mut degrees = vec![0usize; k];
    for (r, row) in et.cells.iter().enumerate() {
        for cell in row.iter().flatten() {
            if cell.is_dmz {
                degrees[r] += 1;
            }
        }
    }
    let dmz_total = degrees.iter().sum();
    degrees.sort();
    RowDegreeDistribution {
        degrees,
        dmz_total,
        k,
    }
}

/// Result of the hide/fill analysis for a single regime.
#[derive(Debug, Clone)]
pub struct HideFillResult {
    /// The regime address.
    pub regime_addr: Vec<u8>,
    /// Number of strut constants in this regime.
    pub n_struts: usize,
    /// DMZ count (same for all struts in regime).
    pub dmz_count: usize,
    /// Sorted row-degree distribution (same for all struts in regime).
    pub row_degrees: Vec<usize>,
    /// Whether this regime is "full fill" (all addressable cells are DMZ).
    pub is_full_fill: bool,
    /// Number of cells in the core (DMZ in ALL struts of this regime).
    pub core_size: usize,
    /// Number of cells in the union (DMZ in ANY strut of this regime).
    pub union_size: usize,
    /// Total addressable cells (K*(K-1) - K = K^2 - 2K, minus strut-opposites).
    pub total_addressable: usize,
}

/// Perform the hide/fill analysis for all regimes at CD level `n`.
///
/// Returns one `HideFillResult` per regime, sorted by regime address.
pub fn hide_fill_analysis(n: usize) -> Vec<HideFillResult> {
    use std::collections::{BTreeMap, BTreeSet};

    let max_s = (1usize << (n - 1)) - 1;
    // Group strut constants by regime address
    let mut regime_struts: BTreeMap<Vec<u8>, Vec<usize>> = BTreeMap::new();
    for s in 1..=max_s {
        let addr = regime_address(n, s);
        regime_struts.entry(addr).or_default().push(s);
    }

    let mut results = Vec::new();

    for (addr, struts) in &regime_struts {
        // Compute row-degree distribution for each strut
        let first_et = create_strutted_et(n, struts[0]);
        let first_dist = row_degree_distribution(&first_et);
        let total_addressable = first_et.total_possible;

        // Verify all struts have same distribution
        for &s in &struts[1..] {
            let et = create_strutted_et(n, s);
            let dist = row_degree_distribution(&et);
            debug_assert_eq!(
                dist, first_dist,
                "N={}, S={}: row-degree differs from S={}",
                n, s, struts[0]
            );
        }

        // Compute core (intersection) and union across all struts
        let sets: Vec<BTreeSet<(usize, usize)>> = struts
            .iter()
            .map(|&s| {
                let et = create_strutted_et(n, s);
                let k = et.tone_row.k;
                let mut set = BTreeSet::new();
                for r in 0..k {
                    for c in 0..k {
                        if let Some(cell) = &et.cells[r][c]
                            && cell.is_dmz
                        {
                            set.insert((r, c));
                        }
                    }
                }
                set
            })
            .collect();

        let core: BTreeSet<_> = sets.iter().skip(1).fold(sets[0].clone(), |acc, s| {
            acc.intersection(s).copied().collect()
        });
        let union: BTreeSet<_> = sets
            .iter()
            .skip(1)
            .fold(sets[0].clone(), |acc, s| acc.union(s).copied().collect());

        let is_full_fill = first_dist.dmz_total == total_addressable;

        results.push(HideFillResult {
            regime_addr: addr.clone(),
            n_struts: struts.len(),
            dmz_count: first_dist.dmz_total,
            row_degrees: first_dist.degrees,
            is_full_fill,
            core_size: core.len(),
            union_size: union.len(),
            total_addressable,
        });
    }

    results
}
