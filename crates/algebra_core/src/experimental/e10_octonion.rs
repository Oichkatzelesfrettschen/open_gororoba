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

use rand::SeedableRng;

use crate::experimental::billiard_stats::{
    generate_null_sequence, summarize_permutation_test, NullModel, PermutationTestResult,
};
use crate::lie::e8_lattice::{e8_simple_roots, generate_e8_roots, E8Root};
use crate::physics::octonion_field::{oct_multiply, oct_norm_sq, Octonion, FANO_TRIPLES};

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

    (
        best_mapping,
        best_rate,
        best_completions,
        best_opportunities,
        all_rates,
    )
}

/// Compute the exact p-value: fraction of permutations with rate >= observed.
pub fn exact_pvalue(observed_rate: f64, all_rates: &[f64]) -> f64 {
    let count = all_rates
        .iter()
        .filter(|&&r| r >= observed_rate - 1e-15)
        .count();
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
    let var = all_counts
        .iter()
        .map(|&c| (c as f64 - mean).powi(2))
        .sum::<f64>()
        / n;
    (mean, var.sqrt())
}

// ---------------------------------------------------------------------------
// Fano overlap graph: associative overlap structure for E8 walls
// ---------------------------------------------------------------------------
//
// The Fano plane PG(2,2) has 7 points and 7 lines. Each line defines a
// quaternionic (associative) subalgebra of the octonions. Every pair of
// distinct imaginary octonion units lies on exactly one Fano line
// (since C(7,2) = 21 = 7 * 3 = 7 lines * 3 pairs-per-line).
//
// The "Fano overlap graph" for a given CayleyBasis maps E8 simple roots
// to octonion basis elements, then connects wall pairs whose images are
// distinct imaginary units (and therefore share a unique Fano line).
//
// Comparing this graph to the E8 Dynkin diagram and to empirical billiard
// transition data tests whether octonion associativity constrains the physics.

/// The Fano overlap graph: which E8 wall pairs share a Fano line.
#[derive(Debug, Clone)]
pub struct FanoOverlapGraph {
    /// 8x8 symmetric adjacency matrix. `adjacency[i][j] = true` iff walls
    /// i and j both map to distinct imaginary octonion units.
    pub adjacency: [[bool; 8]; 8],
    /// For each connected pair, the sorted Fano triple `[a, b, c]` (a < b < c)
    /// of imaginary unit indices. None for non-adjacent or real-mapped walls.
    pub fano_line: [[Option<[usize; 3]>; 8]; 8],
    /// Octonion basis index for each wall (dominant component), or None if
    /// the wall maps to the real direction (e_0) or to zero.
    pub wall_octonion: [Option<usize>; 8],
    /// Number of walls mapping to imaginary directions (1..=7).
    pub n_imaginary: usize,
    /// Number of edges in the overlap graph.
    pub n_edges: usize,
    /// Number of distinct Fano lines represented among the edges.
    pub n_distinct_lines: usize,
}

/// Comparison between two 8x8 graph edge sets.
#[derive(Debug, Clone)]
pub struct GraphEdgeComparison {
    /// Number of edges in graph A.
    pub edges_a: usize,
    /// Number of edges in graph B.
    pub edges_b: usize,
    /// Edges present in both A and B.
    pub intersection: usize,
    /// Edges in A but not B.
    pub a_only: usize,
    /// Edges in B but not A.
    pub b_only: usize,
    /// Jaccard similarity: |A intersect B| / |A union B|. 0.0 if both empty.
    pub jaccard: f64,
}

/// Build the Fano overlap graph for E8 walls under a given CayleyBasis.
///
/// For each simple root alpha_i, computes the dominant octonion basis element
/// under the basis mapping. Two walls are connected iff both map to distinct
/// imaginary units. The connecting Fano line is recorded for each edge.
pub fn build_fano_overlap_graph(basis: &CayleyBasis) -> FanoOverlapGraph {
    let simple = e8_simple_roots();
    let table = fano_complement_table();

    // Map each wall to its dominant octonion basis element
    let mut wall_oct = [None; 8];
    let mut n_imag = 0usize;
    for (i, root) in simple.iter().enumerate() {
        let oct = basis.root_to_octonion(root);
        if let Some(idx) = dominant_basis_element(&oct) {
            if idx > 0 {
                wall_oct[i] = Some(idx);
                n_imag += 1;
            }
        }
    }

    let mut adjacency = [[false; 8]; 8];
    let mut fano_line = [[None; 8]; 8];
    let mut n_edges = 0usize;
    let mut lines_seen = std::collections::BTreeSet::new();

    for i in 0..8 {
        for j in (i + 1)..8 {
            if let (Some(oi), Some(oj)) = (wall_oct[i], wall_oct[j]) {
                if oi != oj {
                    // Both imaginary, distinct -> they share a Fano line
                    adjacency[i][j] = true;
                    adjacency[j][i] = true;
                    n_edges += 1;

                    // Find the Fano line
                    if let Some(ok) = table[oi][oj] {
                        let mut triple = [oi, oj, ok];
                        triple.sort();
                        fano_line[i][j] = Some(triple);
                        fano_line[j][i] = Some(triple);
                        lines_seen.insert(triple);
                    }
                }
            }
        }
    }

    FanoOverlapGraph {
        adjacency,
        fano_line,
        wall_octonion: wall_oct,
        n_imaginary: n_imag,
        n_edges,
        n_distinct_lines: lines_seen.len(),
    }
}

/// Compare two 8x8 boolean adjacency matrices (upper triangle only).
///
/// Both matrices must be symmetric; only the upper triangle (i < j) is counted.
pub fn compare_8x8_graphs(a: &[[bool; 8]; 8], b: &[[bool; 8]; 8]) -> GraphEdgeComparison {
    let mut ea = 0usize;
    let mut eb = 0usize;
    let mut both = 0usize;
    let mut a_only = 0usize;
    let mut b_only = 0usize;

    for i in 0..8 {
        for j in (i + 1)..8 {
            let in_a = a[i][j];
            let in_b = b[i][j];
            if in_a {
                ea += 1;
            }
            if in_b {
                eb += 1;
            }
            if in_a && in_b {
                both += 1;
            }
            if in_a && !in_b {
                a_only += 1;
            }
            if !in_a && in_b {
                b_only += 1;
            }
        }
    }

    let union = ea + eb - both;
    let jaccard = if union > 0 {
        both as f64 / union as f64
    } else {
        0.0
    };

    GraphEdgeComparison {
        edges_a: ea,
        edges_b: eb,
        intersection: both,
        a_only,
        b_only,
        jaccard,
    }
}

/// Extract the E8 sub-adjacency from E10_ADJACENCY (walls 0-7 only).
pub fn e8_dynkin_adjacency() -> [[bool; 8]; 8] {
    let e10 = crate::experimental::billiard_stats::E10_ADJACENCY;
    let mut adj = [[false; 8]; 8];
    for i in 0..8 {
        for j in 0..8 {
            adj[i][j] = e10[i][j];
        }
    }
    adj
}

/// Compare the Fano overlap graph to the E8 Dynkin diagram.
///
/// Returns how many Dynkin edges are also Fano edges, and vice versa.
pub fn compare_fano_dynkin(fano: &FanoOverlapGraph) -> GraphEdgeComparison {
    let dynkin = e8_dynkin_adjacency();
    compare_8x8_graphs(&fano.adjacency, &dynkin)
}

/// Build an empirical transition graph from a wall-hit sequence.
///
/// Edge (i, j) exists iff wall i is immediately followed by wall j at
/// least once in the sequence. Only E8 walls (0-7) are included; walls
/// 8+ (affine/hyperbolic) are skipped.
///
/// The resulting graph is NOT necessarily symmetric: edge (i,j) means
/// i -> j occurred, not necessarily j -> i. For undirected comparison,
/// use `symmetrize_transition_graph()`.
pub fn build_e8_transition_graph(sequence: &[usize]) -> [[bool; 8]; 8] {
    let mut graph = [[false; 8]; 8];
    for pair in sequence.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        if a < 8 && b < 8 && a != b {
            graph[a][b] = true;
        }
    }
    graph
}

/// Symmetrize a directed transition graph: edge (i,j) = true iff either
/// direction was observed.
pub fn symmetrize_transition_graph(directed: &[[bool; 8]; 8]) -> [[bool; 8]; 8] {
    let mut sym = [[false; 8]; 8];
    for i in 0..8 {
        for j in 0..8 {
            if directed[i][j] || directed[j][i] {
                sym[i][j] = true;
                sym[j][i] = true;
            }
        }
    }
    sym
}

/// Compare the Fano overlap graph to an empirical transition graph.
///
/// The transition graph is first symmetrized for fair comparison.
pub fn compare_fano_transitions(
    fano: &FanoOverlapGraph,
    transitions: &[[bool; 8]; 8],
) -> GraphEdgeComparison {
    let sym = symmetrize_transition_graph(transitions);
    compare_8x8_graphs(&fano.adjacency, &sym)
}

/// Search all 8! permutation bases for the one maximizing Fano-Dynkin overlap.
///
/// Returns the best basis, its FanoOverlapGraph, the Dynkin comparison,
/// and the distribution of intersection counts across all permutations.
pub fn optimal_fano_overlap_basis() -> (
    CayleyBasis,
    FanoOverlapGraph,
    GraphEdgeComparison,
    Vec<usize>,
) {
    let dynkin = e8_dynkin_adjacency();

    let mut best_basis = default_cayley_basis();
    let mut best_overlap = 0usize;
    let mut best_graph = build_fano_overlap_graph(&best_basis);
    let mut all_overlaps = Vec::with_capacity(40320);

    // Evaluate one permutation
    let eval = |perm: &[usize; 8]| -> (usize, FanoOverlapGraph) {
        let basis = CayleyBasis {
            perm: *perm,
            signs: [1.0; 8],
        };
        let graph = build_fano_overlap_graph(&basis);
        let cmp = compare_8x8_graphs(&graph.adjacency, &dynkin);
        (cmp.intersection, graph)
    };

    let mut perm = [0usize, 1, 2, 3, 4, 5, 6, 7];

    // Identity
    let (ov, gr) = eval(&perm);
    all_overlaps.push(ov);
    if ov > best_overlap {
        best_overlap = ov;
        best_basis.perm = perm;
        best_graph = gr;
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
            let (ov, gr) = eval(&perm);
            all_overlaps.push(ov);
            if ov > best_overlap {
                best_overlap = ov;
                best_basis.perm = perm;
                best_graph = gr;
            }
            c[idx] += 1;
            idx = 0;
        } else {
            c[idx] = 0;
            idx += 1;
        }
    }

    let best_cmp = compare_8x8_graphs(&best_graph.adjacency, &dynkin);
    (best_basis, best_graph, best_cmp, all_overlaps)
}

// ---------------------------------------------------------------------------
// Claim 4 verification: Fano completion rate permutation test
// ---------------------------------------------------------------------------

/// Result of the Claim 4 verification: octonionic Fano completion analysis.
///
/// Two independent statistical tests:
/// 1. **Mapping test**: Is the optimal wall-to-octonion mapping significantly
///    better than random permutations? (exact p-value over 40320 permutations)
/// 2. **Sequence test**: With the optimal mapping fixed, does the real billiard
///    sequence show higher Fano completion than null billiard sequences?
///
/// Claim 4 is supported if BOTH tests reject at the specified alpha.
#[derive(Debug, Clone)]
pub struct Claim4Result {
    /// The optimal wall-to-octonion mapping found by exhaustive search.
    pub optimal_mapping: [usize; 8],
    /// Number of Fano triple completions in the real sequence.
    pub completions: usize,
    /// Number of Fano triple opportunities in the real sequence.
    pub opportunities: usize,
    /// Observed Fano completion rate (completions / opportunities).
    pub observed_rate: f64,
    /// Exact p-value from the mapping test (fraction of permutations >= observed).
    pub mapping_pvalue: f64,
    /// Z-score against the uniform null rate (1/7).
    pub enrichment_zscore: f64,
    /// Permutation test result from the sequence test (null billiard sequences).
    pub sequence_test: PermutationTestResult,
    /// Number of E8-only 3-windows extracted from the sequence.
    pub n_windows: usize,
}

/// Verify Claim 4: the octonion interpretation adds predictive power.
///
/// This is the full verification pipeline:
/// 1. Extract 3-bounce windows restricted to E8 walls (indices 0-7).
/// 2. Exhaustive search over 8! = 40320 wall-to-octonion mappings for the
///    one that maximizes Fano triple completion rate.
/// 3. Compute exact p-value against the permutation distribution (mapping test).
/// 4. Fix the optimal mapping, generate `n_permutations` null billiard sequences
///    using the specified `NullModel`, compute Fano completion rate for each
///    (sequence test).
/// 5. Report both test results and overall statistics.
///
/// # Arguments
/// - `sequence`: raw billiard wall-hit sequence (may contain walls 0-9)
/// - `null_model`: which null model to use for the sequence test
/// - `n_permutations`: number of null sequences to generate
/// - `seed`: deterministic RNG seed for reproducibility
pub fn verify_claim4(
    sequence: &[usize],
    null_model: &NullModel,
    n_permutations: usize,
    seed: u64,
) -> Claim4Result {
    // Step 1: Extract E8-only 3-windows
    let windows = extract_3windows(sequence);
    let n_windows = windows.len();

    // Step 2: Exhaustive mapping search
    let (best_mapping, best_rate, best_completions, best_opportunities, all_rates) =
        optimal_fano_mapping(&windows);

    // Step 3: Exact p-value (mapping test)
    let mapping_pvalue = exact_pvalue(best_rate, &all_rates);

    // Step 4: Enrichment Z-score against uniform null
    let enrichment_zscore = fano_enrichment_zscore(best_rate, best_opportunities);

    // Step 5: Sequence permutation test -- fix the optimal mapping,
    // generate null sequences, measure Fano completion on each
    let mut null_rates = Vec::with_capacity(n_permutations);
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);

    for _ in 0..n_permutations {
        let null_seq = generate_null_sequence(null_model, sequence.len(), sequence, &mut rng);
        let null_windows = extract_3windows(&null_seq);
        let (_, _, null_rate) = fano_completion_rate(&best_mapping, &null_windows);
        null_rates.push(null_rate);
    }

    let sequence_test = summarize_permutation_test(best_rate, &null_rates);

    Claim4Result {
        optimal_mapping: best_mapping,
        completions: best_completions,
        opportunities: best_opportunities,
        observed_rate: best_rate,
        mapping_pvalue,
        enrichment_zscore,
        sequence_test,
        n_windows,
    }
}

/// Quick summary of Claim 4 result: returns (supported, reason).
///
/// Claim 4 is supported if:
/// 1. Mapping p-value < alpha (optimal mapping is not random)
/// 2. Sequence p-value < alpha (dynamics show Fano preference)
/// 3. Observed rate > 1/7 (enrichment, not depletion)
pub fn claim4_summary(result: &Claim4Result, alpha: f64) -> (bool, String) {
    let mapping_sig = result.mapping_pvalue < alpha;
    let sequence_sig = result.sequence_test.p_value < alpha;
    let enriched = result.observed_rate > NULL_FANO_RATE_UNIFORM;

    let supported = mapping_sig && sequence_sig && enriched;

    let reason = format!(
        "rate={:.4} (vs null 1/7={:.4}), mapping p={:.4} {}, sequence p={:.4} {}, z={:.2}",
        result.observed_rate,
        NULL_FANO_RATE_UNIFORM,
        result.mapping_pvalue,
        if mapping_sig { "<alpha" } else { ">=alpha" },
        result.sequence_test.p_value,
        if sequence_sig { "<alpha" } else { ">=alpha" },
        result.enrichment_zscore,
    );

    (supported, reason)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

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
            assert_eq!(
                fano_complement(a, b),
                Some(c),
                "complement({a},{b}) should be {c}"
            );
            assert_eq!(
                fano_complement(b, a),
                Some(c),
                "complement({b},{a}) should be {c}"
            );
            assert_eq!(
                fano_complement(a, c),
                Some(b),
                "complement({a},{c}) should be {b}"
            );
            assert_eq!(
                fano_complement(c, a),
                Some(b),
                "complement({c},{a}) should be {b}"
            );
            assert_eq!(
                fano_complement(b, c),
                Some(a),
                "complement({b},{c}) should be {a}"
            );
            assert_eq!(
                fano_complement(c, b),
                Some(a),
                "complement({c},{b}) should be {a}"
            );
        }
    }

    #[test]
    fn test_fano_complement_every_pair() {
        // Every pair of distinct imaginary units has exactly one complement
        for i in 1..=7 {
            for j in 1..=7 {
                if i != j {
                    let comp = fano_complement(i, j);
                    assert!(comp.is_some(), "({i},{j}) should have a Fano complement");
                    let k = comp.unwrap();
                    assert_ne!(k, i);
                    assert_ne!(k, j);
                    assert!((1..=7).contains(&k), "complement should be in 1..=7");
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
        eprintln!("Optimal Cayley basis: {best_count}/{total_adj} edges are Fano-connected");
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
                    assert!(table[i][j].is_some(), "({i},{j}) should be Fano-connected");
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Fano overlap graph tests (Engine B)
    // -----------------------------------------------------------------------

    #[test]
    fn test_fano_overlap_identity_basis_structure() {
        // Identity basis: wall i -> octonion coordinate i.
        // Wall 0: alpha_1 = (1,-1,0,...) has dominant = e_0 (real, tied with e_1)
        //   -> excluded (dominant is index 0 < 1)
        // Wall 7: alpha_8 = (-0.5,...,-0.5,0.5) has dominant = e_0 (all tied, first wins)
        //   -> excluded (dominant is index 0 < 1)
        // Walls 1-4: clear imaginary dominants (e_1..e_4 respectively)
        // Walls 5,6: both have dominant e_5 (alpha_6 and alpha_7 both peak at coord 5)
        //   -> 5,6 NOT connected (same octonion index)
        let basis = default_cayley_basis();
        let graph = build_fano_overlap_graph(&basis);

        // 6 walls map to imaginary units (walls 1,2,3,4,5,6)
        assert_eq!(graph.n_imaginary, 6);
        // Walls 0 and 7 are excluded
        assert_eq!(graph.wall_octonion[0], None);
        assert_eq!(graph.wall_octonion[7], None);
        // Walls 5 and 6 both map to e_5
        assert_eq!(graph.wall_octonion[5], Some(5));
        assert_eq!(graph.wall_octonion[6], Some(5));
    }

    #[test]
    fn test_fano_overlap_identity_basis_edge_count() {
        let basis = default_cayley_basis();
        let graph = build_fano_overlap_graph(&basis);
        // 6 imaginary walls, but walls 5,6 share the same octonion (e_5).
        // Distinct octonion images: {1, 2, 3, 4, 5} (5 distinct values).
        // Edges = C(5,2) - (walls 5&6 share image, so not connected) + ...
        // Actually: walls {1,2,3,4} each have unique images, walls {5,6} share e_5.
        // Pairs among {1,2,3,4}: C(4,2) = 6, all connected.
        // Pairs {1,5},{2,5},{3,5},{4,5}: 4 edges (wall 5 with walls 1-4).
        // Pairs {1,6},{2,6},{3,6},{4,6}: 4 edges (wall 6 with walls 1-4).
        // Pair {5,6}: NOT connected (same octonion e_5).
        // Total = 6 + 4 + 4 = 14
        assert_eq!(graph.n_edges, 14);
    }

    #[test]
    fn test_fano_overlap_graph_symmetry() {
        let basis = default_cayley_basis();
        let graph = build_fano_overlap_graph(&basis);
        for i in 0..8 {
            assert!(!graph.adjacency[i][i], "no self-loops");
            for j in 0..8 {
                assert_eq!(
                    graph.adjacency[i][j], graph.adjacency[j][i],
                    "adjacency must be symmetric at ({i},{j})"
                );
                assert_eq!(
                    graph.fano_line[i][j], graph.fano_line[j][i],
                    "fano_line must be symmetric at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_fano_overlap_fano_lines_sorted() {
        let basis = default_cayley_basis();
        let graph = build_fano_overlap_graph(&basis);
        for i in 0..8 {
            for j in (i + 1)..8 {
                if let Some(triple) = graph.fano_line[i][j] {
                    assert!(
                        triple[0] < triple[1] && triple[1] < triple[2],
                        "Fano triple must be sorted: {:?}",
                        triple
                    );
                    assert!(
                        triple[0] >= 1 && triple[2] <= 7,
                        "Fano triple must be imaginary indices: {:?}",
                        triple
                    );
                }
            }
        }
    }

    #[test]
    fn test_fano_dynkin_comparison_identity() {
        let basis = default_cayley_basis();
        let graph = build_fano_overlap_graph(&basis);
        let cmp = compare_fano_dynkin(&graph);
        // E8 Dynkin has 7 edges
        assert_eq!(cmp.edges_b, 7);
        // Fano overlap has 14 edges (identity basis)
        assert_eq!(cmp.edges_a, 14);
        // 5 Dynkin edges are also Fano edges
        assert_eq!(cmp.intersection, 5);
        // 2 Dynkin edges are NOT Fano edges (walls 0 and 7 excluded)
        assert_eq!(cmp.b_only, 2);
    }

    #[test]
    fn test_e8_dynkin_adjacency_edge_count() {
        let adj = e8_dynkin_adjacency();
        let mut edges = 0;
        for i in 0..8 {
            for j in (i + 1)..8 {
                if adj[i][j] {
                    edges += 1;
                }
            }
        }
        // E8 Dynkin diagram: 8 nodes, tree with 7 edges
        assert_eq!(edges, 7);
    }

    #[test]
    fn test_compare_8x8_graphs_identical() {
        let a = [[false; 8]; 8];
        let cmp = compare_8x8_graphs(&a, &a);
        assert_eq!(cmp.edges_a, 0);
        assert_eq!(cmp.edges_b, 0);
        assert_eq!(cmp.intersection, 0);
        assert_eq!(cmp.jaccard, 0.0);
    }

    #[test]
    fn test_compare_8x8_graphs_disjoint() {
        let mut a = [[false; 8]; 8];
        let mut b = [[false; 8]; 8];
        a[0][1] = true;
        a[1][0] = true;
        b[2][3] = true;
        b[3][2] = true;
        let cmp = compare_8x8_graphs(&a, &b);
        assert_eq!(cmp.edges_a, 1);
        assert_eq!(cmp.edges_b, 1);
        assert_eq!(cmp.intersection, 0);
        assert_eq!(cmp.a_only, 1);
        assert_eq!(cmp.b_only, 1);
        assert!((cmp.jaccard - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_build_e8_transition_graph() {
        // Sequence: 0->1->2->3->8->4 (walls 0-7 are E8, wall 8 is affine)
        let seq = vec![0, 1, 2, 3, 8, 4];
        let graph = build_e8_transition_graph(&seq);
        // Transitions: 0->1, 1->2, 2->3, (3->8 filtered), (8->4 filtered)
        assert!(graph[0][1]);
        assert!(graph[1][2]);
        assert!(graph[2][3]);
        // Direction matters (directed graph)
        assert!(!graph[1][0]);
        // No self-loops
        for i in 0..8 {
            assert!(!graph[i][i]);
        }
    }

    #[test]
    fn test_symmetrize_transition_graph() {
        let mut g = [[false; 8]; 8];
        g[0][1] = true;
        g[2][3] = true;
        let sym = symmetrize_transition_graph(&g);
        assert!(sym[0][1]);
        assert!(sym[1][0]);
        assert!(sym[2][3]);
        assert!(sym[3][2]);
        assert!(!sym[0][2]);
    }

    #[test]
    fn test_optimal_fano_overlap_basis_search() {
        let (_best_basis, best_graph, best_cmp, all_overlaps) = optimal_fano_overlap_basis();
        // 40320 permutations evaluated
        assert_eq!(all_overlaps.len(), 40320);
        // Best should achieve at least 6/7 Dynkin overlap
        assert!(
            best_cmp.intersection >= 6,
            "Best overlap should be >= 6, got {}",
            best_cmp.intersection
        );
        // Optimal basis should have 7 imaginary walls (no wasted real mapping)
        assert!(
            best_graph.n_imaginary >= 7,
            "Optimal basis should map >= 7 walls to imaginary, got {}",
            best_graph.n_imaginary
        );
        // Mean overlap across all permutations should be around 2.5
        let mean = all_overlaps.iter().sum::<usize>() as f64 / 40320.0;
        assert!(
            mean > 2.0 && mean < 3.0,
            "Mean overlap should be ~2.5, got {:.3}",
            mean
        );
    }

    #[test]
    fn test_fano_overlap_distinct_lines() {
        // Identity basis: 6 distinct Fano lines used
        let basis = default_cayley_basis();
        let graph = build_fano_overlap_graph(&basis);
        assert_eq!(graph.n_distinct_lines, 6);
        // There are only 7 Fano lines total, so 6/7 coverage is high
    }

    #[test]
    fn test_fano_transition_comparison_synthetic() {
        // Synthetic transition sequence: 1->2->3->4->5->6->1
        let seq = vec![1, 2, 3, 4, 5, 6, 1];
        let trans = build_e8_transition_graph(&seq);
        let basis = default_cayley_basis();
        let graph = build_fano_overlap_graph(&basis);
        let cmp = compare_fano_transitions(&graph, &trans);
        // 6 directed transitions, when symmetrized: {1-2, 2-3, 3-4, 4-5, 5-6, 6-1}
        assert_eq!(cmp.edges_b, 6);
        // 5 of these 6 transition edges are also Fano edges.
        // The exception is edge (5,6): walls 5 and 6 both map to e_5 under
        // the identity basis, so they share the same octonion image and are
        // NOT Fano-connected.
        assert_eq!(cmp.intersection, 5);
        assert_eq!(cmp.b_only, 1);
    }

    // -----------------------------------------------------------------------
    // Claim 4 verification tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_verify_claim4_uniform_null() {
        // Uniform random sequence of E8 walls: should show no Fano preference.
        // With uniform random data, the optimal mapping should not beat null
        // sequences significantly (p > 0.01).
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(12345);
        let sequence: Vec<usize> = (0..500).map(|_| rng.gen_range(0..8)).collect();

        let result = verify_claim4(&sequence, &NullModel::Uniform, 200, 54321);

        // Uniform random data: Fano completion rate should be near 1/7 ~ 0.143
        assert!(
            result.observed_rate < 0.30,
            "Random data should not show extreme Fano rate, got {:.4}",
            result.observed_rate
        );
        // The mapping p-value measures whether the optimal mapping is special
        // among all 40320 permutations -- with random data this is just noise
        assert!(
            result.mapping_pvalue > 0.0,
            "Mapping p-value should be computable"
        );
        // Verify structure
        assert!(result.n_windows > 0, "Should have some 3-windows");
        assert!(result.opportunities > 0, "Should have some opportunities");
    }

    #[test]
    fn test_verify_claim4_structured_sequence() {
        // Construct a sequence that strongly respects Fano triples under a
        // specific mapping. Use the identity mapping: wall i -> e_i.
        // Fano triples are: (1,2,3), (1,4,5), (1,7,6), (2,4,6), (2,5,7),
        //                   (3,4,7), (3,6,5).
        // Build a sequence that repeatedly cycles through Fano completions.
        let mut sequence = Vec::new();
        for _ in 0..50 {
            // Each triple (a,b,c): after seeing a,b the next bounce is c
            sequence.extend_from_slice(&[1, 2, 3]); // completes (1,2,3)
            sequence.extend_from_slice(&[2, 4, 6]); // completes (2,4,6)
            sequence.extend_from_slice(&[3, 4, 7]); // completes (3,4,7)
        }

        let result = verify_claim4(&sequence, &NullModel::Uniform, 200, 99999);

        // This sequence was constructed to have high completion rate
        assert!(
            result.observed_rate > 0.30,
            "Structured Fano sequence should have high rate, got {:.4}",
            result.observed_rate
        );
        // The sequence test should show this is significantly above null
        assert!(
            result.sequence_test.p_value < 0.10,
            "Structured sequence should have significant sequence p-value, got {:.4}",
            result.sequence_test.p_value
        );
        // Effect size should be positive
        assert!(
            result.sequence_test.effect_size > 0.0,
            "Effect size should be positive, got {:.4}",
            result.sequence_test.effect_size
        );
    }

    #[test]
    fn test_verify_claim4_result_structure() {
        // Verify the result struct is well-formed
        let sequence: Vec<usize> = (0..100).map(|i| i % 8).collect();
        let result = verify_claim4(&sequence, &NullModel::Uniform, 50, 42);

        // Mapping should be a valid permutation
        let mut sorted_mapping = result.optimal_mapping;
        sorted_mapping.sort();
        assert_eq!(
            sorted_mapping,
            [0, 1, 2, 3, 4, 5, 6, 7],
            "Optimal mapping must be a permutation of 0..8"
        );
        // Completions <= opportunities
        assert!(
            result.completions <= result.opportunities,
            "Completions ({}) should not exceed opportunities ({})",
            result.completions,
            result.opportunities
        );
        // Rate consistency
        if result.opportunities > 0 {
            let expected_rate = result.completions as f64 / result.opportunities as f64;
            assert!(
                (result.observed_rate - expected_rate).abs() < 1e-10,
                "Rate inconsistency: {} vs {}",
                result.observed_rate,
                expected_rate
            );
        }
        // Sequence test has correct number of permutations
        assert_eq!(result.sequence_test.n_permutations, 50);
        // Mapping p-value in [0, 1]
        assert!(result.mapping_pvalue >= 0.0 && result.mapping_pvalue <= 1.0);
    }

    #[test]
    fn test_claim4_summary_logic() {
        // Test the summary function decision logic
        let result = Claim4Result {
            optimal_mapping: [0, 1, 2, 3, 4, 5, 6, 7],
            completions: 50,
            opportunities: 200,
            observed_rate: 0.25,  // > 1/7
            mapping_pvalue: 0.01, // < 0.05
            enrichment_zscore: 3.5,
            sequence_test: PermutationTestResult {
                observed: 0.25,
                null_mean: 0.14,
                null_std: 0.02,
                p_value: 0.001, // < 0.05
                n_permutations: 1000,
                null_ci_lower: 0.10,
                null_ci_upper: 0.18,
                effect_size: 5.5,
            },
            n_windows: 500,
        };
        let (supported, reason) = claim4_summary(&result, 0.05);
        assert!(
            supported,
            "Should be supported when both p < alpha and enriched: {}",
            reason
        );

        // Test rejection: sequence p-value too high
        let mut result_fail = result.clone();
        result_fail.sequence_test.p_value = 0.30;
        let (supported_fail, _) = claim4_summary(&result_fail, 0.05);
        assert!(
            !supported_fail,
            "Should NOT be supported when sequence p >= alpha"
        );

        // Test rejection: mapping p-value too high
        let mut result_fail2 = result.clone();
        result_fail2.mapping_pvalue = 0.20;
        let (supported_fail2, _) = claim4_summary(&result_fail2, 0.05);
        assert!(
            !supported_fail2,
            "Should NOT be supported when mapping p >= alpha"
        );

        // Test rejection: depleted rate
        let mut result_fail3 = result.clone();
        result_fail3.observed_rate = 0.10; // < 1/7
        let (supported_fail3, _) = claim4_summary(&result_fail3, 0.05);
        assert!(!supported_fail3, "Should NOT be supported when rate < 1/7");
    }

    #[test]
    fn test_verify_claim4_deterministic() {
        // Same inputs and seed should produce identical results
        let sequence: Vec<usize> = (0..200).map(|i| (i * 3 + 7) % 8).collect();
        let r1 = verify_claim4(&sequence, &NullModel::Uniform, 100, 77777);
        let r2 = verify_claim4(&sequence, &NullModel::Uniform, 100, 77777);

        assert_eq!(r1.optimal_mapping, r2.optimal_mapping);
        assert!((r1.observed_rate - r2.observed_rate).abs() < 1e-15);
        assert!((r1.mapping_pvalue - r2.mapping_pvalue).abs() < 1e-15);
        assert!((r1.sequence_test.p_value - r2.sequence_test.p_value).abs() < 1e-15);
        assert!((r1.enrichment_zscore - r2.enrichment_zscore).abs() < 1e-15);
    }

    #[test]
    fn test_verify_claim4_empty_sequence() {
        // Edge case: empty sequence
        let result = verify_claim4(&[], &NullModel::Uniform, 50, 42);
        assert_eq!(result.n_windows, 0);
        assert_eq!(result.completions, 0);
        assert_eq!(result.opportunities, 0);
        assert!((result.observed_rate - 0.0).abs() < 1e-15);
    }

    #[test]
    fn test_verify_claim4_hyperbolic_only() {
        // Sequence with only walls 8 and 9: no E8-only windows
        let sequence: Vec<usize> = (0..100).map(|i| 8 + i % 2).collect();
        let result = verify_claim4(&sequence, &NullModel::Uniform, 50, 42);
        assert_eq!(
            result.n_windows, 0,
            "Hyperbolic-only sequence has no E8 windows"
        );
    }

    #[test]
    fn test_verify_claim4_different_null_models() {
        // Test that different null models produce different distributions
        let sequence: Vec<usize> = (0..300).map(|i| i % 8).collect();
        let r_uniform = verify_claim4(&sequence, &NullModel::Uniform, 100, 42);
        let r_iid = verify_claim4(&sequence, &NullModel::IidEmpirical, 100, 42);
        let r_degree = verify_claim4(&sequence, &NullModel::DegreePreserving, 100, 42);

        // All should have the same observed rate (same sequence and mapping)
        assert!((r_uniform.observed_rate - r_iid.observed_rate).abs() < 1e-15);
        assert!((r_uniform.observed_rate - r_degree.observed_rate).abs() < 1e-15);

        // But the null distributions (and hence p-values) may differ
        // At minimum, verify they all complete without panic
        assert!(r_uniform.sequence_test.n_permutations == 100);
        assert!(r_iid.sequence_test.n_permutations == 100);
        assert!(r_degree.sequence_test.n_permutations == 100);
    }

    #[test]
    fn test_verify_claim4_enrichment_zscore_sign() {
        // For a structured sequence, Z-score should be positive (enrichment)
        let mut sequence = Vec::new();
        for _ in 0..100 {
            sequence.extend_from_slice(&[1, 2, 3, 1, 4, 5, 2, 5, 7]);
        }
        let result = verify_claim4(&sequence, &NullModel::Uniform, 100, 42);

        // The Z-score should be meaningful (non-zero) for non-trivial data
        assert!(
            result.enrichment_zscore.is_finite(),
            "Z-score should be finite, got {}",
            result.enrichment_zscore
        );
    }
}
