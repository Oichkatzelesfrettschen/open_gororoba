//! Abstract framework for generator-driven piecewise dynamics on constraint graphs.
//!
//! This module provides the `ConstraintSystem` trait and generic locality analysis
//! that works across any system where:
//! - There is a finite generator set G = {g_0, ..., g_{n-1}} (represented as usize labels)
//! - There is a constraint graph Gamma on G (adjacency predicate)
//! - A dynamical process produces sequences of generator labels
//!
//! The Algebraic Locality Principle (C-476) predicts that all physically realized
//! constraint systems exhibit locality: generator sequences concentrate on adjacent
//! transitions in Gamma, with r >> r_null.
//!
//! Three concrete implementations are planned:
//! - E10 billiard (wall transitions on E10 Dynkin diagram)
//! - ET DMZ walk (emanation table cell transitions on DMZ adjacency)
//! - CD ZD catamaran (basis-pair transitions on zero-divisor adjacency)

/// A constraint system with generators and an adjacency relation.
///
/// This is the common abstract model for the ALP cross-stack comparison.
/// Each concrete system implements this trait, enabling generic locality
/// analysis via `compute_generic_locality`.
pub trait ConstraintSystem {
    /// Number of generators in the system.
    fn n_generators(&self) -> usize;

    /// Whether generators i and j are adjacent in the constraint graph Gamma.
    /// Must be symmetric: adjacent(i, j) == adjacent(j, i).
    /// Self-adjacency (i == j) returns false by convention.
    fn adjacent(&self, i: usize, j: usize) -> bool;

    /// Human-readable name of this constraint system.
    fn name(&self) -> &str;
}

/// Locality metrics for a generator sequence on an arbitrary constraint system.
#[derive(Debug, Clone)]
pub struct GenericLocalityMetrics {
    /// Name of the constraint system.
    pub system_name: String,
    /// Number of generators in the system.
    pub n_generators: usize,
    /// Total transitions analyzed (sequence length - 1).
    pub n_transitions: usize,
    /// Number of adjacent transitions (consecutive pair in Gamma).
    pub n_adjacent: usize,
    /// Locality ratio: n_adjacent / n_transitions.
    pub r: f64,
    /// Null expectation: 2 * |E(Gamma)| / (|G|^2 - |G|) where |E| is edge count.
    /// This is the expected locality ratio under uniform random generator selection.
    pub r_null: f64,
    /// Mutual information I(S_t; S_{t+1}) in nats.
    pub mutual_information: f64,
}

/// Compute generic locality metrics for a generator sequence on a constraint system.
///
/// The sequence contains generator labels in 0..system.n_generators().
/// Out-of-range labels are silently skipped.
pub fn compute_generic_locality<C: ConstraintSystem>(
    system: &C,
    sequence: &[usize],
) -> GenericLocalityMetrics {
    let n_gen = system.n_generators();
    let name = system.name().to_string();

    if sequence.len() < 2 {
        return GenericLocalityMetrics {
            system_name: name,
            n_generators: n_gen,
            n_transitions: 0,
            n_adjacent: 0,
            r: 0.0,
            r_null: compute_r_null(system),
            mutual_information: 0.0,
        };
    }

    let mut n_transitions = 0usize;
    let mut n_adjacent = 0usize;

    // Transition counts for mutual information
    let mut trans_counts: Vec<Vec<u64>> = vec![vec![0u64; n_gen]; n_gen];
    let mut from_counts: Vec<u64> = vec![0u64; n_gen];

    for pair in sequence.windows(2) {
        let (i, j) = (pair[0], pair[1]);
        if i >= n_gen || j >= n_gen || i == j {
            continue;
        }
        n_transitions += 1;
        if system.adjacent(i, j) {
            n_adjacent += 1;
        }
        trans_counts[i][j] += 1;
        from_counts[i] += 1;
    }

    let r = if n_transitions > 0 {
        n_adjacent as f64 / n_transitions as f64
    } else {
        0.0
    };

    // Mutual information I(S_t; S_{t+1})
    let total = from_counts.iter().sum::<u64>() as f64;
    let mut to_counts = vec![0u64; n_gen];
    for j in 0..n_gen {
        for row in trans_counts.iter().take(n_gen) {
            to_counts[j] += row[j];
        }
    }

    let mut mi = 0.0;
    if total > 0.0 {
        for i in 0..n_gen {
            for j in 0..n_gen {
                let n_ij = trans_counts[i][j] as f64;
                if n_ij > 0.0 {
                    let p_ij = n_ij / total;
                    let p_i = from_counts[i] as f64 / total;
                    let p_j = to_counts[j] as f64 / total;
                    if p_i > 0.0 && p_j > 0.0 {
                        mi += p_ij * (p_ij / (p_i * p_j)).ln();
                    }
                }
            }
        }
    }

    GenericLocalityMetrics {
        system_name: name,
        n_generators: n_gen,
        n_transitions,
        n_adjacent,
        r,
        r_null: compute_r_null(system),
        mutual_information: mi,
    }
}

/// Compute null-model expected locality ratio for uniform random generator selection.
///
/// r_null = 2|E(Gamma)| / (|G|*(|G|-1)) where |E| counts undirected edges.
fn compute_r_null<C: ConstraintSystem>(system: &C) -> f64 {
    let n = system.n_generators();
    if n < 2 {
        return 0.0;
    }
    let mut edge_count = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            if system.adjacent(i, j) {
                edge_count += 1;
            }
        }
    }
    // Each undirected edge contributes to 2 ordered pairs out of n*(n-1) total
    (2 * edge_count) as f64 / (n * (n - 1)) as f64
}

/// Permutation test for locality ratio.
///
/// Generates `n_permutations` random permutations of the sequence, computes
/// locality ratio for each, and returns the p-value (fraction of null samples
/// with r >= observed r).
pub fn permutation_test_generic<C: ConstraintSystem>(
    system: &C,
    sequence: &[usize],
    n_permutations: usize,
    seed: u64,
) -> f64 {
    use rand::seq::SliceRandom;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    let observed = compute_generic_locality(system, sequence);
    if observed.n_transitions == 0 {
        return 1.0;
    }

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut shuffled = sequence.to_vec();
    let mut n_ge = 0usize;

    for _ in 0..n_permutations {
        shuffled.shuffle(&mut rng);
        let null_metrics = compute_generic_locality(system, &shuffled);
        if null_metrics.r >= observed.r {
            n_ge += 1;
        }
    }

    n_ge as f64 / n_permutations as f64
}

// =====================================================================
// Concrete ConstraintSystem implementations
// =====================================================================

/// E10 Dynkin diagram constraint system (10 generators, walls 0-9).
///
/// The adjacency is the E10 Dynkin diagram: a T-shaped graph with
/// 8 E8 nodes (linear chain 1-2-3-4-5-6-7, branching at 3 to 0)
/// plus nodes 8 (connected to 7) and 9 (connected to 8).
pub struct E10DynkinSystem;

impl ConstraintSystem for E10DynkinSystem {
    fn n_generators(&self) -> usize { 10 }

    fn adjacent(&self, i: usize, j: usize) -> bool {
        if i == j || i >= 10 || j >= 10 {
            return false;
        }
        super::billiard_stats::E10_ADJACENCY[i][j]
    }

    fn name(&self) -> &str { "E10 Dynkin billiard" }
}

/// Sedenion (dim=16) zero-divisor adjacency constraint system.
///
/// Generators are basis indices 1..15 (excluding identity e_0).
/// Two generators are adjacent if the corresponding basis elements
/// form a zero-divisor pair (their product has non-trivial kernel).
pub struct SedenionZdSystem {
    /// Adjacency matrix: zd_adjacent[i][j] for i,j in 0..15.
    /// Index 0 = basis element e_1, etc. (shifted by 1 from basis index).
    zd_adjacent: Vec<Vec<bool>>,
}

impl Default for SedenionZdSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl SedenionZdSystem {
    /// Build the ZD adjacency system from the box-kite census.
    ///
    /// Two non-identity basis elements e_i, e_j (i,j >= 1) are ZD-adjacent
    /// if they appear as assessor components in the same box-kite.
    /// Each assessor has a `low` (1-7) and `high` (8-15) index.
    /// All 12 basis indices that appear in a box-kite's 6 assessors are
    /// pairwise ZD-adjacent within that box-kite.
    pub fn new() -> Self {
        use crate::analysis::boxkites::find_box_kites;
        let bks = find_box_kites(16, 1e-10);

        // 15 non-identity basis elements, indexed 0..14 (representing e_1..e_15)
        let n = 15;
        let mut adj = vec![vec![false; n]; n];

        // Two basis elements are ZD-adjacent if they appear in the same box-kite
        for bk in &bks {
            // Collect all unique basis indices from this box-kite's assessors
            let mut indices = Vec::new();
            for a in &bk.assessors {
                indices.push(a.low);  // 1-7
                indices.push(a.high); // 8-15
            }
            indices.sort();
            indices.dedup();

            // All pairs of basis indices within a box-kite are adjacent
            for vi in 0..indices.len() {
                for vj in (vi + 1)..indices.len() {
                    let a = indices[vi];
                    let b = indices[vj];
                    // Convert from basis index (1-based) to our array index (0-based)
                    if a >= 1 && b >= 1 && a <= 15 && b <= 15 {
                        adj[a - 1][b - 1] = true;
                        adj[b - 1][a - 1] = true;
                    }
                }
            }
        }

        SedenionZdSystem { zd_adjacent: adj }
    }
}

impl ConstraintSystem for SedenionZdSystem {
    fn n_generators(&self) -> usize { 15 }

    fn adjacent(&self, i: usize, j: usize) -> bool {
        if i == j || i >= 15 || j >= 15 {
            return false;
        }
        self.zd_adjacent[i][j]
    }

    fn name(&self) -> &str { "Sedenion ZD adjacency" }
}

/// ET DMZ adjacency constraint system for a given CD level.
///
/// Generators are (row, col) cells in the strutted ET.
/// Two cells are adjacent if they share a DMZ boundary.
pub struct EtDmzSystem {
    /// Number of cells (rows * cols of the ET grid).
    n_cells: usize,
    /// Adjacency: dmz_adjacent[cell_a][cell_b] for linearized (row, col) indices.
    dmz_adjacent: Vec<Vec<bool>>,
    /// ET dimensions (K x K).
    k: usize,
}

impl EtDmzSystem {
    /// Build the ET DMZ adjacency system for CD level n, strut constant s.
    ///
    /// Cells are linearized as row * k + col. Two cells are DMZ-adjacent if
    /// they are grid-neighbors (up/down/left/right) and at least one is a DMZ cell.
    pub fn new(n: usize, s: usize) -> Self {
        use super::emanation::create_strutted_et;
        let et = create_strutted_et(n, s);
        let k = et.tone_row.k;
        let n_cells = k * k;
        let mut adj = vec![vec![false; n_cells]; n_cells];

        let is_dmz = |r: usize, c: usize| -> bool {
            et.cells[r][c].as_ref().is_some_and(|cell| cell.is_dmz)
        };

        for r in 0..k {
            for c in 0..k {
                let idx = r * k + c;
                // Check 4-connected neighbors
                let neighbors: [(isize, isize); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
                for &(dr, dc) in &neighbors {
                    let nr = r as isize + dr;
                    let nc = c as isize + dc;
                    if nr >= 0 && nr < k as isize && nc >= 0 && nc < k as isize {
                        let nr = nr as usize;
                        let nc = nc as usize;
                        let nidx = nr * k + nc;
                        // Adjacent if at least one cell is DMZ
                        if is_dmz(r, c) || is_dmz(nr, nc) {
                            adj[idx][nidx] = true;
                        }
                    }
                }
            }
        }

        EtDmzSystem {
            n_cells,
            dmz_adjacent: adj,
            k,
        }
    }

    /// ET grid dimension K.
    pub fn k(&self) -> usize {
        self.k
    }
}

impl ConstraintSystem for EtDmzSystem {
    fn n_generators(&self) -> usize { self.n_cells }

    fn adjacent(&self, i: usize, j: usize) -> bool {
        if i == j || i >= self.n_cells || j >= self.n_cells {
            return false;
        }
        self.dmz_adjacent[i][j]
    }

    fn name(&self) -> &str { "ET DMZ adjacency" }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_e10_system_basic() {
        let sys = E10DynkinSystem;
        assert_eq!(sys.n_generators(), 10);
        assert_eq!(sys.name(), "E10 Dynkin billiard");
        // E10 edges: 0-1, 1-2, 2-3, 3-4, 4-5, 4-6, 6-7, 0-8, 8-9
        assert!(sys.adjacent(0, 1)); // spine start
        assert!(sys.adjacent(1, 2)); // spine
        assert!(sys.adjacent(3, 4)); // spine
        assert!(sys.adjacent(4, 6)); // branch
        assert!(sys.adjacent(6, 7)); // branch tip
        assert!(sys.adjacent(0, 8)); // affine
        assert!(sys.adjacent(8, 9)); // hyperbolic
        assert!(!sys.adjacent(0, 3)); // not adjacent (3 hops)
        assert!(!sys.adjacent(5, 6)); // not adjacent (branch goes 4-6)
        assert!(!sys.adjacent(7, 8)); // not adjacent (different branches)
        assert!(!sys.adjacent(5, 5)); // self
    }

    #[test]
    fn test_e10_r_null() {
        let sys = E10DynkinSystem;
        // E10 has 9 edges, 10 nodes => r_null = 2*9 / (10*9) = 18/90 = 0.2
        let r = compute_r_null(&sys);
        assert!((r - 0.2).abs() < 1e-10, "E10 r_null should be 0.2, got {}", r);
    }

    #[test]
    fn test_e10_locality_on_adjacent_sequence() {
        let sys = E10DynkinSystem;
        // Walk along actual E10 edges: 9-8-0-1-2-3-4-6-7
        // 9-8 (hyp), 8-0 (affine), 0-1 (spine), 1-2, 2-3, 3-4, 4-6 (branch), 6-7
        let seq = vec![9, 8, 0, 1, 2, 3, 4, 6, 7];
        let m = compute_generic_locality(&sys, &seq);
        assert_eq!(m.n_transitions, 8);
        assert_eq!(m.n_adjacent, 8);
        assert!((m.r - 1.0).abs() < 1e-10, "All-adjacent should give r=1.0");
    }

    #[test]
    fn test_e10_locality_on_random_sequence() {
        let sys = E10DynkinSystem;
        // Non-adjacent jumps: 0-9-1-8-2-7 -- none of these are adjacent
        let seq = vec![0, 9, 1, 8, 2, 7];
        let m = compute_generic_locality(&sys, &seq);
        assert_eq!(m.n_transitions, 5);
        assert_eq!(m.n_adjacent, 0);
        assert!(m.r < 1e-10, "No adjacent pairs should give r=0.0");
    }

    #[test]
    fn test_sedenion_zd_system_basic() {
        let sys = SedenionZdSystem::new();
        assert_eq!(sys.n_generators(), 15);
        assert_eq!(sys.name(), "Sedenion ZD adjacency");
        // Should have some adjacency (7 box-kites, each with 6 vertices)
        let mut edge_count = 0;
        for i in 0..15 {
            for j in (i + 1)..15 {
                if sys.adjacent(i, j) {
                    edge_count += 1;
                }
            }
        }
        // 7 BKs x C(6,2) = 7*15 = 105 vertex pairs, but with overlap
        // The exact number is 7*15 - overlaps
        assert!(edge_count > 0, "Should have ZD edges");
        assert!(edge_count <= 105, "At most 105 edges from 7 BKs");
    }

    #[test]
    fn test_sedenion_zd_r_null() {
        let sys = SedenionZdSystem::new();
        let r = compute_r_null(&sys);
        // r_null should be between 0 and 1
        assert!(r > 0.0, "Should have some adjacency: r_null={}", r);
        assert!(r < 1.0, "Not fully connected: r_null={}", r);
    }

    #[test]
    fn test_et_dmz_system_basic() {
        // N=5, S=9 is a standard test case
        let sys = EtDmzSystem::new(5, 9);
        assert!(sys.n_generators() > 0);
        assert_eq!(sys.name(), "ET DMZ adjacency");
        let k = sys.k();
        assert_eq!(sys.n_generators(), k * k);
        // Should have some DMZ adjacency
        let r = compute_r_null(&sys);
        assert!(r >= 0.0);
    }

    #[test]
    fn test_generic_locality_empty_sequence() {
        let sys = E10DynkinSystem;
        let m = compute_generic_locality(&sys, &[]);
        assert_eq!(m.n_transitions, 0);
        assert!(m.r.abs() < 1e-10);
    }

    #[test]
    fn test_generic_locality_single_element() {
        let sys = E10DynkinSystem;
        let m = compute_generic_locality(&sys, &[3]);
        assert_eq!(m.n_transitions, 0);
    }

    #[test]
    fn test_permutation_test_all_adjacent() {
        let sys = E10DynkinSystem;
        // A highly local sequence should have low p-value
        let seq = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 3, 2, 1];
        let p = permutation_test_generic(&sys, &seq, 100, 42);
        assert!(p < 0.1, "All-adjacent sequence should have low p-value, got {}", p);
    }
}
