//! E10-Octonion bridge: mapping E8 billiard walls to octonion basis elements.
//!
//! The E10 Kac-Moody algebra contains E8 as a sub-root system (walls 0-7).
//! The E8 root system is connected to the octonions through the Cayley integer
//! construction (Conway-Sloane). This module tests whether the billiard wall
//! transition sequence respects the Fano plane structure of the octonion
//! multiplication table.
//!
//! # Predictive claim (Claim 4)
//!
//! If walls 0-7 map to octonion basis elements via some permutation sigma,
//! then for 3-consecutive-bounce windows (w_a, w_b, w_c) where sigma(w_a)
//! and sigma(w_b) are distinct imaginary units, the third element sigma(w_c)
//! should complete the Fano triple more often than chance.
//!
//! # Method
//!
//! 1. Exhaustive search over all 8! = 40320 permutations of {0..7} -> {e_0..e_7}
//! 2. For each permutation, compute "Fano completion rate": fraction of
//!    3-windows where the third bounce completes the Fano triple
//! 3. Compare optimal rate to distribution of all permutations (exact p-value)
//!
//! # Literature
//! - Conway & Sloane: "Sphere Packings, Lattices and Groups", Ch. 8
//! - Wilson: "The Finite Simple Groups", Sec. 4.3 (E8 and octonions)
//! - Baez: "The Octonions", Bull. AMS 39 (2002), Sec. 4.4

use crate::e8_lattice::{e8_simple_roots, generate_e8_roots, E8Root};
use crate::octonion_field::{oct_multiply, oct_norm_sq, Octonion, FANO_TRIPLES};

/// Given two distinct imaginary octonion indices (1..=7), return the third
/// index that completes their unique Fano line, or None if the input is
/// invalid (real unit, same index, or out of range).
pub fn fano_complement(i: usize, j: usize) -> Option<usize> {
    if i == 0 || j == 0 || i == j || i > 7 || j > 7 {
        return None;
    }
    for &(a, b, c) in FANO_TRIPLES.iter() {
        // FANO_TRIPLES are oriented: e_a * e_b = +e_c (cyclic).
        // The unordered LINE is {a, b, c}.
        let pts = [a, b, c];
        if pts.contains(&i) && pts.contains(&j) {
            for &p in &pts {
                if p != i && p != j {
                    return Some(p);
                }
            }
        }
    }
    None
}

/// Build the full Fano complement lookup table for efficiency.
///
/// Returns a 8x8 array where table[i][j] = Some(k) if {e_i, e_j, e_k}
/// is a Fano triple, or None otherwise.
pub fn fano_complement_table() -> [[Option<usize>; 8]; 8] {
    let mut table = [[None; 8]; 8];
    for (i, row) in table.iter_mut().enumerate().skip(1).take(7) {
        for (j, cell) in row.iter_mut().enumerate().skip(1).take(7) {
            if i != j {
                *cell = fano_complement(i, j);
            }
        }
    }
    table
}

/// Extract consecutive 3-bounce windows from a wall-hit sequence.
///
/// Only includes windows where all three walls are in the E8 range (0..8).
pub fn extract_3windows(sequence: &[usize]) -> Vec<(usize, usize, usize)> {
    sequence
        .windows(3)
        .filter(|w| w[0] < 8 && w[1] < 8 && w[2] < 8)
        .map(|w| (w[0], w[1], w[2]))
        .collect()
}

/// Compute the Fano triple completion rate for a given wall-to-octonion mapping.
///
/// `mapping[wall]` = octonion basis index (0..8). Must be a permutation of {0..7}.
///
/// Returns `(completions, opportunities, rate)`:
/// - `opportunities`: windows where sigma(w_a) and sigma(w_b) are distinct
///   imaginary units (both in 1..=7). Since every pair of distinct imaginary
///   octonions lies on exactly one Fano line, this is always well-defined.
/// - `completions`: subset where sigma(w_c) equals the Fano complement.
/// - `rate` = completions / opportunities (0.0 if no opportunities).
pub fn fano_completion_rate(
    mapping: &[usize; 8],
    windows: &[(usize, usize, usize)],
) -> (usize, usize, f64) {
    let table = fano_complement_table();
    let mut completions = 0usize;
    let mut opportunities = 0usize;

    for &(wa, wb, wc) in windows {
        let oa = mapping[wa];
        let ob = mapping[wb];
        let oc = mapping[wc];

        // Both must be imaginary (non-zero) and distinct
        if oa == 0 || ob == 0 || oa == ob {
            continue;
        }

        if let Some(complement) = table[oa][ob] {
            opportunities += 1;
            if oc == complement {
                completions += 1;
            }
        }
    }

    let rate = if opportunities > 0 {
        completions as f64 / opportunities as f64
    } else {
        0.0
    };
    (completions, opportunities, rate)
}

/// Search over all 8! = 40320 permutations to find the mapping that maximizes
/// the Fano triple completion rate.
///
/// Returns `(best_mapping, best_rate, best_completions, best_opportunities, all_rates)`.
/// `all_rates` contains the rate for every permutation (for computing exact p-value).
pub fn optimal_fano_mapping(
    windows: &[(usize, usize, usize)],
) -> ([usize; 8], f64, usize, usize, Vec<f64>) {
    let mut best_mapping = [0usize; 8];
    let mut best_rate = -1.0f64;
    let mut best_completions = 0;
    let mut best_opportunities = 0;
    let mut all_rates = Vec::with_capacity(40320);

    let mut perm = [0, 1, 2, 3, 4, 5, 6, 7];

    // Evaluate identity permutation first
    let (comp, opp, rate) = fano_completion_rate(&perm, windows);
    all_rates.push(rate);
    if rate > best_rate {
        best_rate = rate;
        best_completions = comp;
        best_opportunities = opp;
        best_mapping = perm;
    }

    // Heap's algorithm for generating all permutations in-place
    let mut c = [0usize; 8];
    let mut i = 0;
    while i < 8 {
        if c[i] < i {
            if i % 2 == 0 {
                perm.swap(0, i);
            } else {
                perm.swap(c[i], i);
            }

            let (comp, opp, rate) = fano_completion_rate(&perm, windows);
            all_rates.push(rate);
            if rate > best_rate {
                best_rate = rate;
                best_completions = comp;
                best_opportunities = opp;
                best_mapping = perm;
            }

            c[i] += 1;
            i = 0;
        } else {
            c[i] = 0;
            i += 1;
        }
    }

    (best_mapping, best_rate, best_completions, best_opportunities, all_rates)
}

/// Compute the exact p-value: fraction of permutations with rate >= observed.
pub fn exact_pvalue(observed_rate: f64, all_rates: &[f64]) -> f64 {
    let count = all_rates.iter().filter(|&&r| r >= observed_rate - 1e-15).count();
    count as f64 / all_rates.len() as f64
}

/// Null expectation for Fano completion rate under uniform random transitions.
///
/// If the third bounce is uniform among the 7 remaining E8 walls (excluding
/// the current wall), the probability of hitting the specific Fano complement
/// is 1/7.
pub const NULL_FANO_RATE_UNIFORM: f64 = 1.0 / 7.0;

/// Compute the Fano enrichment Z-score against the uniform null.
///
/// Under H0, completions ~ Binomial(n, 1/7).
/// Z = (rate - 1/7) / sqrt((1/7)(6/7) / n)
pub fn fano_enrichment_zscore(rate: f64, opportunities: usize) -> f64 {
    if opportunities == 0 {
        return 0.0;
    }
    let n = opportunities as f64;
    let p = NULL_FANO_RATE_UNIFORM;
    let se = (p * (1.0 - p) / n).sqrt();
    if se < 1e-15 {
        return 0.0;
    }
    (rate - p) / se
}

/// Describe the Fano plane adjacency structure implied by a mapping.
///
/// Returns a string representation showing which E8 walls form Fano triples.
pub fn describe_fano_structure(mapping: &[usize; 8]) -> Vec<(usize, usize, usize)> {
    let mut triples = Vec::new();
    let inv = invert_mapping(mapping);

    for &(a, b, c) in FANO_TRIPLES.iter() {
        // Map octonion indices back to wall indices
        if let (Some(&wa), Some(&wb), Some(&wc)) = (inv.get(&a), inv.get(&b), inv.get(&c)) {
            triples.push((wa, wb, wc));
        }
    }
    triples
}

/// Invert a mapping: octonion_index -> wall_index.
fn invert_mapping(mapping: &[usize; 8]) -> std::collections::HashMap<usize, usize> {
    mapping
        .iter()
        .enumerate()
        .map(|(wall, &oct)| (oct, wall))
        .collect()
}

// ---------------------------------------------------------------------------
// Cayley integer <-> E8 root bridge
// ---------------------------------------------------------------------------

/// A coordinate permutation mapping E8 root coordinates to octonion
/// basis elements.
///
/// The E8 simple roots in Bourbaki coordinates live in R^8. The Cayley
/// integers (integral octonions) also live in an 8D space with basis
/// {1, e_1, ..., e_7}. This struct records an explicit identification
/// between the two bases.
///
/// `perm[i]` = octonion basis index (0=real, 1..7=imaginary) for the
/// i-th coordinate of an E8 root vector.
#[derive(Debug, Clone)]
pub struct CayleyBasis {
    /// Permutation: E8 coordinate index -> octonion basis index.
    pub perm: [usize; 8],
    /// Sign flips: +1 or -1 for each coordinate.
    pub signs: [f64; 8],
}

impl CayleyBasis {
    /// Apply this basis to convert an E8 root into a Cayley integer (octonion).
    pub fn root_to_octonion(&self, root: &E8Root) -> Octonion {
        let mut oct = [0.0; 8];
        for (i, &c) in root.coords.iter().enumerate() {
            oct[self.perm[i]] += self.signs[i] * c;
        }
        oct
    }
}

/// Default Cayley basis: identity permutation with no sign flips.
///
/// This directly identifies E8 coordinate i with octonion basis e_i
/// (where e_0 = 1, the real unit).
pub fn default_cayley_basis() -> CayleyBasis {
    CayleyBasis {
        perm: [0, 1, 2, 3, 4, 5, 6, 7],
        signs: [1.0; 8],
    }
}

/// Verify that all 240 E8 roots, when mapped through a CayleyBasis,
/// produce valid Cayley integers (octonions of norm 1).
///
/// Returns `(valid_count, max_norm_error)`.
pub fn verify_cayley_integer_norms(basis: &CayleyBasis) -> (usize, f64) {
    let roots = generate_e8_roots();
    let mut valid = 0;
    let mut max_err = 0.0f64;
    for root in &roots {
        let oct = basis.root_to_octonion(root);
        let norm_sq = oct_norm_sq(&oct);
        let err = (norm_sq - 2.0).abs();
        if err < 1e-10 {
            valid += 1;
        }
        max_err = max_err.max(err);
    }
    (valid, max_err)
}

/// For a given CayleyBasis, compute the multiplication table of the
/// 8 simple roots viewed as Cayley integers.
///
/// Returns an 8x8 array where entry [i][j] is the octonion product
/// of the i-th and j-th simple root images. This reveals which simple
/// root pairs are related by Fano triple structure.
pub fn simple_root_products(basis: &CayleyBasis) -> [[Octonion; 8]; 8] {
    let simple = e8_simple_roots();
    let octs: Vec<Octonion> = simple.iter().map(|r| basis.root_to_octonion(r)).collect();
    let mut table = [[[0.0; 8]; 8]; 8];
    for i in 0..8 {
        for j in 0..8 {
            table[i][j] = oct_multiply(&octs[i], &octs[j]);
        }
    }
    table
}

/// Check whether Dynkin adjacency of E8 simple roots corresponds to
/// Fano triple membership when roots are mapped to octonion basis
/// elements via a CayleyBasis.
///
/// Two simple roots alpha_i and alpha_j are Dynkin-adjacent iff
/// <alpha_i, alpha_j> = -1 (off-diagonal Cartan matrix entry is -1).
///
/// Two octonion basis elements e_a and e_b are Fano-connected iff
/// there exists a Fano triple containing both.
///
/// This function computes what fraction of Dynkin-adjacent pairs map
/// to Fano-connected octonion elements under the given basis.
///
/// Returns `(adjacent_fano_count, total_adjacent, fano_rate)`.
pub fn dynkin_fano_correspondence(basis: &CayleyBasis) -> (usize, usize, f64) {
    let simple = e8_simple_roots();
    let octs: Vec<Octonion> = simple.iter().map(|r| basis.root_to_octonion(r)).collect();
    let table = fano_complement_table();

    let mut adj_fano = 0usize;
    let mut total_adj = 0usize;

    for i in 0..8 {
        for j in (i + 1)..8 {
            let ip = simple[i].inner_product(&simple[j]);
            if (ip + 1.0).abs() < 1e-10 {
                // Dynkin-adjacent pair
                total_adj += 1;

                // Find which basis elements the octonions are closest to.
                // Each simple root maps to a specific octonion via the basis.
                // We need the "dominant" basis element for each.
                let dom_i = dominant_basis_element(&octs[i]);
                let dom_j = dominant_basis_element(&octs[j]);

                if let (Some(di), Some(dj)) = (dom_i, dom_j) {
                    if di > 0 && dj > 0 && di != dj && table[di][dj].is_some() {
                        adj_fano += 1;
                    }
                }
            }
        }
    }

    let rate = if total_adj > 0 {
        adj_fano as f64 / total_adj as f64
    } else {
        0.0
    };
    (adj_fano, total_adj, rate)
}

/// Find the octonion basis element with the largest absolute component.
///
/// For E8 simple roots that are pure coordinate vectors (like alpha_1 = (1,-1,0,...)),
/// this identifies which basis direction dominates. Returns None if the octonion
/// is zero or has all very small components.
fn dominant_basis_element(oct: &Octonion) -> Option<usize> {
    let mut best_idx = 0;
    let mut best_val = 0.0f64;
    for (i, &v) in oct.iter().enumerate() {
        if v.abs() > best_val {
            best_val = v.abs();
            best_idx = i;
        }
    }
    if best_val < 1e-10 {
        None
    } else {
        Some(best_idx)
    }
}

/// Search over all 8! permutations and 2^8 sign patterns to find the
/// CayleyBasis that maximizes Dynkin-Fano correspondence.
///
/// Since 8! * 2^8 = 10,321,920 is expensive, we search only over the
/// 8! permutations with identity signs first. Sign-searching can be
/// added later if needed.
///
/// Returns `(best_basis, best_fano_count, total_adjacent, all_fano_counts)`.
pub fn optimal_cayley_basis() -> (CayleyBasis, usize, usize, Vec<usize>) {
    let simple = e8_simple_roots();
    let table = fano_complement_table();

    // Precompute Dynkin-adjacent pairs
    let mut adj_pairs = Vec::new();
    for i in 0..8 {
        for j in (i + 1)..8 {
            let ip = simple[i].inner_product(&simple[j]);
            if (ip + 1.0).abs() < 1e-10 {
                adj_pairs.push((i, j));
            }
        }
    }
    let total_adj = adj_pairs.len();

    // Precompute root coordinates
    let coords: Vec<[f64; 8]> = simple.iter().map(|r| r.coords).collect();

    let mut best_basis = default_cayley_basis();
    let mut best_count = 0usize;
    let mut all_counts = Vec::with_capacity(40320);

    let mut perm = [0usize, 1, 2, 3, 4, 5, 6, 7];

    let eval = |perm: &[usize; 8]| -> usize {
        let mut count = 0;
        for &(i, j) in &adj_pairs {
            // Map simple root i through this permutation
            let mut oct_i = [0.0f64; 8];
            let mut oct_j = [0.0f64; 8];
            for (k, &c) in coords[i].iter().enumerate() {
                oct_i[perm[k]] += c;
            }
            for (k, &c) in coords[j].iter().enumerate() {
                oct_j[perm[k]] += c;
            }
            let di = dominant_basis_element(&oct_i);
            let dj = dominant_basis_element(&oct_j);
            if let (Some(a), Some(b)) = (di, dj) {
                if a > 0 && b > 0 && a != b && table[a][b].is_some() {
                    count += 1;
                }
            }
        }
        count
    };

    // Identity permutation
    let count = eval(&perm);
    all_counts.push(count);
    if count > best_count {
        best_count = count;
        best_basis.perm = perm;
    }

    // Heap's algorithm
    let mut c = [0usize; 8];
    let mut idx = 0;
    while idx < 8 {
        if c[idx] < idx {
            if idx % 2 == 0 {
                perm.swap(0, idx);
            } else {
                perm.swap(c[idx], idx);
            }
            let count = eval(&perm);
            all_counts.push(count);
            if count > best_count {
                best_count = count;
                best_basis.perm = perm;
            }
            c[idx] += 1;
            idx = 0;
        } else {
            c[idx] = 0;
            idx += 1;
        }
    }

    (best_basis, best_count, total_adj, all_counts)
}

/// Compute the null expectation for Dynkin-Fano correspondence under
/// a random permutation.
///
/// Given `n_adj` Dynkin-adjacent pairs and `n_fano = 21` Fano-connected
/// pairs (out of C(7,2) = 21 imaginary pairs -- actually ALL imaginary
/// pairs are Fano-connected), the null probability depends on how
/// many simple roots map to imaginary vs real.
pub fn dynkin_fano_null_summary(all_counts: &[usize]) -> (f64, f64) {
    let n = all_counts.len() as f64;
    let mean = all_counts.iter().sum::<usize>() as f64 / n;
    let var = all_counts.iter()
        .map(|&c| (c as f64 - mean).powi(2))
        .sum::<f64>() / n;
    (mean, var.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fano_complement_basic() {
        // From FANO_TRIPLES: (1,2,3) -> complement of (1,2) is 3
        assert_eq!(fano_complement(1, 2), Some(3));
        assert_eq!(fano_complement(2, 1), Some(3));
        assert_eq!(fano_complement(1, 3), Some(2));
        assert_eq!(fano_complement(2, 3), Some(1));
    }

    #[test]
    fn test_fano_complement_all_triples() {
        // Verify all 7 Fano triples produce correct complements
        let expected: [(usize, usize, usize); 7] = [
            (1, 2, 3),
            (1, 4, 5),
            (1, 7, 6),
            (2, 4, 6),
            (2, 5, 7),
            (3, 4, 7),
            (3, 6, 5),
        ];

        for (a, b, c) in expected {
            assert_eq!(fano_complement(a, b), Some(c), "complement({a},{b}) should be {c}");
            assert_eq!(fano_complement(b, a), Some(c), "complement({b},{a}) should be {c}");
            assert_eq!(fano_complement(a, c), Some(b), "complement({a},{c}) should be {b}");
            assert_eq!(fano_complement(c, a), Some(b), "complement({c},{a}) should be {b}");
            assert_eq!(fano_complement(b, c), Some(a), "complement({b},{c}) should be {a}");
            assert_eq!(fano_complement(c, b), Some(a), "complement({c},{b}) should be {a}");
        }
    }

    #[test]
    fn test_fano_complement_every_pair() {
        // Every pair of distinct imaginary units has exactly one complement
        for i in 1..=7 {
            for j in 1..=7 {
                if i != j {
                    let comp = fano_complement(i, j);
                    assert!(
                        comp.is_some(),
                        "({i},{j}) should have a Fano complement"
                    );
                    let k = comp.unwrap();
                    assert_ne!(k, i);
                    assert_ne!(k, j);
                    assert!(k >= 1 && k <= 7, "complement should be in 1..=7");
                }
            }
        }
    }

    #[test]
    fn test_fano_complement_invalid() {
        // Real unit e_0 has no Fano complement
        assert_eq!(fano_complement(0, 1), None);
        assert_eq!(fano_complement(1, 0), None);
        assert_eq!(fano_complement(0, 0), None);
        // Same index
        assert_eq!(fano_complement(3, 3), None);
        // Out of range
        assert_eq!(fano_complement(8, 1), None);
    }

    #[test]
    fn test_fano_complement_table_consistent() {
        let table = fano_complement_table();
        for i in 1..=7 {
            for j in 1..=7 {
                if i != j {
                    assert_eq!(table[i][j], fano_complement(i, j));
                } else {
                    assert_eq!(table[i][j], None);
                }
            }
        }
        // Row/col 0 should all be None
        for k in 0..8 {
            assert_eq!(table[0][k], None);
            assert_eq!(table[k][0], None);
        }
    }

    #[test]
    fn test_extract_3windows() {
        let seq = vec![0, 1, 2, 3, 8, 4, 5];
        let windows = extract_3windows(&seq);
        // (0,1,2), (1,2,3) are valid; (2,3,8) filtered (8 >= 8); (3,8,4) filtered; (8,4,5) filtered
        assert_eq!(windows.len(), 2);
        assert_eq!(windows[0], (0, 1, 2));
        assert_eq!(windows[1], (1, 2, 3));
    }

    #[test]
    fn test_fano_completion_known_sequence() {
        // Identity mapping: wall i -> e_i
        let mapping = [0, 1, 2, 3, 4, 5, 6, 7];
        // Sequence that SHOULD complete a Fano triple:
        // Wall 1 -> e_1, Wall 2 -> e_2, Wall 3 -> e_3
        // Fano triple: (1,2,3), so this is a completion
        let windows = vec![(1, 2, 3)];
        let (comp, opp, rate) = fano_completion_rate(&mapping, &windows);
        assert_eq!(opp, 1);
        assert_eq!(comp, 1);
        assert!((rate - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_fano_completion_non_completing() {
        let mapping = [0, 1, 2, 3, 4, 5, 6, 7];
        // Wall 1 -> e_1, Wall 2 -> e_2, Wall 4 -> e_4
        // Fano complement of (e_1, e_2) is e_3, but e_4 != e_3
        let windows = vec![(1, 2, 4)];
        let (comp, opp, _rate) = fano_completion_rate(&mapping, &windows);
        assert_eq!(opp, 1);
        assert_eq!(comp, 0);
    }

    #[test]
    fn test_fano_completion_real_unit_excluded() {
        let mapping = [0, 1, 2, 3, 4, 5, 6, 7];
        // Wall 0 -> e_0 (real unit), so (0, 1, 2) has no opportunity
        let windows = vec![(0, 1, 2)];
        let (_, opp, _) = fano_completion_rate(&mapping, &windows);
        assert_eq!(opp, 0);
    }

    #[test]
    fn test_optimal_mapping_exhaustive() {
        // With a very short sequence, verify the search runs and returns valid mapping
        let windows = vec![(1, 2, 3), (1, 4, 5), (2, 4, 6), (3, 4, 7)];
        let (best_mapping, best_rate, _comp, _opp, all_rates) = optimal_fano_mapping(&windows);

        // Identity mapping should give 100% completion since all windows are Fano triples
        assert!((best_rate - 1.0).abs() < 1e-10, "best_rate = {best_rate}");
        // Should be the identity mapping or an equivalent
        assert_eq!(best_mapping[0], 0, "e_0 should map to wall 0");

        // Verify all 8! permutations were evaluated
        assert_eq!(all_rates.len(), 40320);
    }

    #[test]
    fn test_exact_pvalue() {
        let all_rates = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        // 0.3 has 3 values >= 0.3 (0.3, 0.4, 0.5), so p = 3/5 = 0.6
        let p = exact_pvalue(0.3, &all_rates);
        assert!((p - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_enrichment_zscore() {
        // 50% completion rate with 100 opportunities
        let z = fano_enrichment_zscore(0.5, 100);
        // Z = (0.5 - 1/7) / sqrt((1/7)(6/7)/100)
        //   = 0.35714 / 0.03499 ~ 10.2
        assert!(z > 9.0, "Z = {z}");
        assert!(z < 11.0, "Z = {z}");
    }

    #[test]
    fn test_describe_fano_structure() {
        let mapping = [0, 1, 2, 3, 4, 5, 6, 7];
        let triples = describe_fano_structure(&mapping);
        // Should recover the 7 original Fano triples (mapped back through identity)
        assert_eq!(triples.len(), 7);
    }

    // -----------------------------------------------------------------------
    // Cayley integer bridge tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_default_cayley_basis_norms() {
        // With identity basis, all 240 E8 roots should have octonion norm^2 = 2
        // (same as root norm^2 since basis is orthonormal).
        let basis = default_cayley_basis();
        let (valid, max_err) = verify_cayley_integer_norms(&basis);
        assert_eq!(valid, 240, "all 240 roots should have norm^2=2");
        assert!(max_err < 1e-10, "max norm error = {max_err}");
    }

    #[test]
    fn test_root_to_octonion_simple() {
        let basis = default_cayley_basis();
        let simple = e8_simple_roots();

        // alpha_1 = (1, -1, 0, 0, 0, 0, 0, 0)
        // With identity basis: oct = [1, -1, 0, 0, 0, 0, 0, 0]
        // = 1*e_0 + (-1)*e_1 = 1 - e_1
        let oct = basis.root_to_octonion(&simple[0]);
        assert!((oct[0] - 1.0).abs() < 1e-10);
        assert!((oct[1] + 1.0).abs() < 1e-10);
        for k in 2..8 {
            assert!(oct[k].abs() < 1e-10);
        }
    }

    #[test]
    fn test_e8_has_7_dynkin_edges() {
        // E8 Dynkin diagram has 7 edges (8 nodes, tree)
        let simple = e8_simple_roots();
        let mut edges = 0;
        for i in 0..8 {
            for j in (i + 1)..8 {
                let ip = simple[i].inner_product(&simple[j]);
                if (ip + 1.0).abs() < 1e-10 {
                    edges += 1;
                }
            }
        }
        assert_eq!(edges, 7, "E8 Dynkin diagram has exactly 7 edges");
    }

    #[test]
    fn test_dominant_basis_element() {
        let oct = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert_eq!(dominant_basis_element(&oct), Some(1));

        let oct2 = [0.0, 0.0, 0.0, 0.0, 0.0, -3.0, 0.0, 0.0];
        assert_eq!(dominant_basis_element(&oct2), Some(5));

        let oct3 = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
        // All equal: returns first (index 0)
        assert_eq!(dominant_basis_element(&oct3), Some(0));

        let zero = [0.0; 8];
        assert_eq!(dominant_basis_element(&zero), None);
    }

    #[test]
    fn test_dynkin_fano_identity_basis() {
        // With identity basis, measure the correspondence
        let basis = default_cayley_basis();
        let (adj_fano, total_adj, rate) = dynkin_fano_correspondence(&basis);
        // E8 has 7 edges. The correspondence rate depends on how the
        // Bourbaki coordinates align with octonion Fano triples.
        assert_eq!(total_adj, 7, "E8 has 7 Dynkin edges");
        // Record the actual rate (this is an empirical measurement)
        eprintln!("Identity basis: {adj_fano}/{total_adj} = {rate:.3}");
    }

    #[test]
    fn test_optimal_cayley_basis_search() {
        // Run the exhaustive search over 8! permutations
        let (best_basis, best_count, total_adj, all_counts) = optimal_cayley_basis();
        assert_eq!(total_adj, 7);
        assert_eq!(all_counts.len(), 40320);

        // The best basis should achieve at least as many as the identity basis
        let identity_count = all_counts[0]; // first entry is identity
        assert!(best_count >= identity_count);

        // Report
        let (mean, std) = dynkin_fano_null_summary(&all_counts);
        eprintln!(
            "Optimal Cayley basis: {best_count}/{total_adj} edges are Fano-connected"
        );
        eprintln!("  perm: {:?}", best_basis.perm);
        eprintln!("  null distribution: mean={mean:.3}, std={std:.3}");
        eprintln!("  identity basis: {identity_count}/{total_adj}");
    }

    #[test]
    fn test_simple_root_products_norm() {
        // Product of two norm-sqrt(2) octonions should have norm <= 2
        // (since |ab| = |a||b| for octonions)
        let basis = default_cayley_basis();
        let table = simple_root_products(&basis);
        for i in 0..8 {
            for j in 0..8 {
                let prod = &table[i][j];
                let norm_sq: f64 = prod.iter().map(|x| x * x).sum();
                // |a*b|^2 = |a|^2 * |b|^2 = 2 * 2 = 4
                assert!(
                    (norm_sq - 4.0).abs() < 1e-10,
                    "product norm^2 should be 4, got {norm_sq} for ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_all_imag_pairs_fano_connected() {
        // In the Fano plane, every pair of distinct imaginary units lies
        // on exactly one line. So ALL pairs are Fano-connected.
        let table = fano_complement_table();
        for i in 1..=7 {
            for j in 1..=7 {
                if i != j {
                    assert!(
                        table[i][j].is_some(),
                        "({i},{j}) should be Fano-connected"
                    );
                }
            }
        }
    }
}
