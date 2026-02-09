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
pub fn compute_generic_locality<C: ConstraintSystem + ?Sized>(
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
fn compute_r_null<C: ConstraintSystem + ?Sized>(system: &C) -> f64 {
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
pub fn permutation_test_generic<C: ConstraintSystem + ?Sized>(
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

/// Twist-product navigation constraint system for sedenion box-kites.
///
/// Generators are the 7 box-kite strut constants (odd numbers 1..13).
/// Two struts are adjacent if they are connected by a twist operation
/// (H* or V* in de Marrais's notation).
pub struct TwistNavigationSystem {
    /// 7 generators, indexed 0..6 (mapping to strut constants 1,3,5,7,9,11,13).
    strut_to_index: Vec<Option<usize>>,
    /// Adjacency matrix: twist_adjacent[i][j] if struts are twist-connected.
    twist_adjacent: [[bool; 7]; 7],
    /// Strut constants in order.
    struts: [usize; 7],
}

impl TwistNavigationSystem {
    /// Build from the Twisted Sisters graph.
    pub fn new() -> Self {
        use super::emanation::twisted_sisters_graph;
        let edges = twisted_sisters_graph();

        let struts = [1, 3, 5, 7, 9, 11, 13];
        let mut strut_to_index = vec![None; 14];
        for (idx, &s) in struts.iter().enumerate() {
            strut_to_index[s] = Some(idx);
        }

        let mut adj = [[false; 7]; 7];
        for e in &edges {
            if let (Some(i), Some(j)) = (strut_to_index.get(e.from_strut).copied().flatten(),
                                          strut_to_index.get(e.to_strut).copied().flatten()) {
                adj[i][j] = true;
                adj[j][i] = true; // symmetrize for undirected ConstraintSystem
            }
        }

        TwistNavigationSystem {
            strut_to_index,
            twist_adjacent: adj,
            struts,
        }
    }

    /// Map strut constant to generator index.
    pub fn strut_index(&self, strut: usize) -> Option<usize> {
        self.strut_to_index.get(strut).copied().flatten()
    }

    /// Get the strut constant for generator index.
    pub fn strut_constant(&self, index: usize) -> Option<usize> {
        self.struts.get(index).copied()
    }
}

impl Default for TwistNavigationSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl ConstraintSystem for TwistNavigationSystem {
    fn n_generators(&self) -> usize { 7 }

    fn adjacent(&self, i: usize, j: usize) -> bool {
        if i == j || i >= 7 || j >= 7 {
            return false;
        }
        self.twist_adjacent[i][j]
    }

    fn name(&self) -> &str { "Twist navigation (Twisted Sisters)" }
}

// =====================================================================
// Walk generators: produce generator sequences by simulating navigation
// =====================================================================

/// Generate a random walk on a constraint graph.
///
/// At each step, with probability `p_adjacent`, pick a random neighbor
/// (if one exists). Otherwise, pick a uniformly random generator.
/// This models "dynamics with local bias" -- the core ALP test.
///
/// Returns a sequence of generator labels of length `n_steps`.
pub fn random_walk_on_graph<C: ConstraintSystem + ?Sized>(
    system: &C,
    n_steps: usize,
    p_adjacent: f64,
    seed: u64,
) -> Vec<usize> {
    use rand::prelude::*;
    use rand_chacha::ChaCha8Rng;

    let n = system.n_generators();
    if n == 0 || n_steps == 0 {
        return Vec::new();
    }

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut seq = Vec::with_capacity(n_steps);

    // Start at a random generator
    let mut current = rng.gen_range(0..n);
    seq.push(current);

    // Precompute neighbor lists for efficiency
    let neighbors: Vec<Vec<usize>> = (0..n)
        .map(|i| (0..n).filter(|&j| system.adjacent(i, j)).collect())
        .collect();

    for _ in 1..n_steps {
        if !neighbors[current].is_empty() && rng.gen_bool(p_adjacent.clamp(0.0, 1.0)) {
            // Move to a random neighbor
            current = neighbors[current][rng.gen_range(0..neighbors[current].len())];
        } else {
            // Jump to a random generator (excluding self)
            loop {
                let next = rng.gen_range(0..n);
                if next != current {
                    current = next;
                    break;
                }
            }
        }
        seq.push(current);
    }
    seq
}

/// Generate a pure uniform random sequence (null model).
///
/// Each step picks a uniformly random generator, with no self-transitions.
pub fn uniform_random_sequence<C: ConstraintSystem + ?Sized>(
    system: &C,
    n_steps: usize,
    seed: u64,
) -> Vec<usize> {
    random_walk_on_graph(system, n_steps, 0.0, seed)
}

/// Generate a pure neighbor-following walk (maximal locality).
///
/// At each step, always pick a random neighbor. If isolated (no neighbors),
/// jump to a random generator.
pub fn neighbor_walk<C: ConstraintSystem + ?Sized>(
    system: &C,
    n_steps: usize,
    seed: u64,
) -> Vec<usize> {
    random_walk_on_graph(system, n_steps, 1.0, seed)
}

/// Result of a cross-stack locality comparison.
#[derive(Debug, Clone)]
pub struct CrossStackComparison {
    /// Metrics for each constraint system under neighbor walk.
    pub walk_metrics: Vec<GenericLocalityMetrics>,
    /// Metrics for each system under uniform random (null).
    pub null_metrics: Vec<GenericLocalityMetrics>,
    /// Permutation p-values for each system.
    pub p_values: Vec<f64>,
}

/// Run the Experiment A cross-stack locality comparison.
///
/// For each constraint system, generate a neighbor-biased walk and a uniform
/// null, compute locality metrics, and run a permutation test.
pub fn cross_stack_comparison(
    systems: &[&dyn ConstraintSystem],
    n_steps: usize,
    n_permutations: usize,
    seed: u64,
) -> CrossStackComparison {
    let mut walk_metrics = Vec::new();
    let mut null_metrics = Vec::new();
    let mut p_values = Vec::new();

    for (i, sys) in systems.iter().enumerate() {
        let sys_seed = seed.wrapping_add(i as u64 * 1000);

        // Generate a neighbor walk (models dynamics with local bias)
        let walk_seq = neighbor_walk(*sys, n_steps, sys_seed);
        let wm = compute_generic_locality(*sys, &walk_seq);

        // Generate a uniform random sequence (null)
        let null_seq = uniform_random_sequence(*sys, n_steps, sys_seed + 500);
        let nm = compute_generic_locality(*sys, &null_seq);

        // Permutation test on the walk sequence
        let p = permutation_test_generic(*sys, &walk_seq, n_permutations, sys_seed + 999);

        walk_metrics.push(wm);
        null_metrics.push(nm);
        p_values.push(p);
    }

    CrossStackComparison {
        walk_metrics,
        null_metrics,
        p_values,
    }
}

// =====================================================================
// ET Discrete Billiard (Experiment B)
// =====================================================================

/// Direction of motion on the ET grid (4-connected cardinal directions).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BilliardDirection {
    Up,    // row decreasing
    Down,  // row increasing
    Left,  // col decreasing
    Right, // col increasing
}

impl BilliardDirection {
    /// Rotate 90 degrees clockwise (used when reflecting off walls).
    fn rotate_cw(self) -> Self {
        match self {
            Self::Up => Self::Right,
            Self::Right => Self::Down,
            Self::Down => Self::Left,
            Self::Left => Self::Up,
        }
    }

    /// Row/col offset for this direction.
    fn delta(self) -> (isize, isize) {
        match self {
            Self::Up => (-1, 0),
            Self::Down => (1, 0),
            Self::Left => (0, -1),
            Self::Right => (0, 1),
        }
    }

    /// Reverse direction (180-degree turn).
    fn reverse(self) -> Self {
        match self {
            Self::Up => Self::Down,
            Self::Down => Self::Up,
            Self::Left => Self::Right,
            Self::Right => Self::Left,
        }
    }

    /// All four directions.
    fn all() -> [Self; 4] {
        [Self::Up, Self::Right, Self::Down, Self::Left]
    }
}

/// Cell type for symbolic dynamics recording.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BilliardCellType {
    /// DMZ (filled) cell.
    Dmz,
    /// Non-DMZ cell (exists but not a zero-divisor pair).
    NonDmz,
}

/// Result of a single billiard trajectory on a specific ET grid.
#[derive(Debug, Clone)]
pub struct EtBilliardTrajectory {
    /// CD level N.
    pub n: usize,
    /// Strut constant S.
    pub s: usize,
    /// Grid dimension K.
    pub k: usize,
    /// Number of simulation steps.
    pub n_steps: usize,
    /// Symbolic dynamics: sequence of cell types visited.
    pub symbolic_dynamics: Vec<BilliardCellType>,
    /// Number of distinct cells visited.
    pub n_distinct_cells: usize,
    /// Cell coverage: n_distinct_cells / n_valid_cells.
    pub coverage: f64,
    /// Shannon entropy rate of bigram distribution (nats per step).
    pub entropy_rate: f64,
    /// Number of direction changes (reflections).
    pub n_reflections: usize,
    /// Mean free path: average steps between reflections.
    pub mean_free_path: f64,
    /// DMZ transition rate: fraction of steps that cross DMZ<->non-DMZ boundary.
    pub dmz_transition_rate: f64,
    /// Number of valid (non-absent) cells in the ET grid.
    pub n_valid_cells: usize,
    /// Number of DMZ cells.
    pub n_dmz_cells: usize,
    /// Fill ratio: n_dmz_cells / n_valid_cells.
    pub fill_ratio: f64,
}

/// Simulate a discrete billiard on the ET grid for a given (N, S).
///
/// The particle moves on valid (non-absent) cells of the K x K ET grid.
/// It reflects when hitting grid boundaries or absent cells.
/// The symbolic dynamics records whether each visited cell is DMZ or non-DMZ.
///
/// Reflection rule: try clockwise rotation. If the forward cell is blocked
/// (out-of-bounds or absent), rotate 90 degrees CW and try again, up to
/// 4 attempts (full rotation). If all 4 directions are blocked, stay in place
/// and reverse direction.
pub fn simulate_et_billiard(
    n: usize,
    s: usize,
    n_steps: usize,
    seed: u64,
) -> EtBilliardTrajectory {
    use super::emanation::create_strutted_et;
    use rand::prelude::*;
    use rand_chacha::ChaCha8Rng;

    let et = create_strutted_et(n, s);
    let k = et.tone_row.k;

    // Build cell type grid: None = absent, Some(true) = DMZ, Some(false) = non-DMZ
    let mut cell_types: Vec<Vec<Option<bool>>> = Vec::with_capacity(k);
    let mut valid_cells: Vec<(usize, usize)> = Vec::new();
    let mut n_dmz = 0usize;

    for r in 0..k {
        let mut row = Vec::with_capacity(k);
        for c in 0..k {
            match &et.cells[r][c] {
                Some(cell) => {
                    row.push(Some(cell.is_dmz));
                    valid_cells.push((r, c));
                    if cell.is_dmz {
                        n_dmz += 1;
                    }
                }
                None => row.push(None),
            }
        }
        cell_types.push(row);
    }

    let n_valid = valid_cells.len();
    if n_valid == 0 || n_steps == 0 {
        return EtBilliardTrajectory {
            n, s, k,
            n_steps: 0,
            symbolic_dynamics: Vec::new(),
            n_distinct_cells: 0,
            coverage: 0.0,
            entropy_rate: 0.0,
            n_reflections: 0,
            mean_free_path: 0.0,
            dmz_transition_rate: 0.0,
            n_valid_cells: n_valid,
            n_dmz_cells: n_dmz,
            fill_ratio: if n_valid > 0 { n_dmz as f64 / n_valid as f64 } else { 0.0 },
        };
    }

    // Pick a random starting position among valid cells
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let start_idx = rng.gen_range(0..n_valid);
    let (mut row, mut col) = valid_cells[start_idx];
    let mut dir = BilliardDirection::all()[rng.gen_range(0..4)];

    // Tracking
    let mut symbolic = Vec::with_capacity(n_steps);
    let mut visited = std::collections::HashSet::new();
    let mut n_reflections = 0usize;
    let mut n_dmz_transitions = 0usize;

    // Record starting cell
    let start_is_dmz = cell_types[row][col].unwrap_or(false);
    symbolic.push(if start_is_dmz { BilliardCellType::Dmz } else { BilliardCellType::NonDmz });
    visited.insert((row, col));

    // Helper: check if (r, c) is a valid (non-absent) cell
    let is_valid = |r: isize, c: isize| -> bool {
        r >= 0 && r < k as isize && c >= 0 && c < k as isize
            && cell_types[r as usize][c as usize].is_some()
    };

    for _ in 1..n_steps {
        // Try to move in the current direction
        let (dr, dc) = dir.delta();
        let nr = row as isize + dr;
        let nc = col as isize + dc;

        if is_valid(nr, nc) {
            // Move succeeds
            row = nr as usize;
            col = nc as usize;
        } else {
            // Hit a wall: try clockwise rotations
            n_reflections += 1;
            let mut moved = false;
            let mut try_dir = dir.rotate_cw();
            for _ in 0..3 {
                let (tdr, tdc) = try_dir.delta();
                let tr = row as isize + tdr;
                let tc = col as isize + tdc;
                if is_valid(tr, tc) {
                    dir = try_dir;
                    row = tr as usize;
                    col = tc as usize;
                    moved = true;
                    break;
                }
                try_dir = try_dir.rotate_cw();
            }
            if !moved {
                // All directions blocked: reverse and stay
                dir = dir.reverse();
            }
        }

        // Record symbolic dynamics
        let is_dmz = cell_types[row][col].unwrap_or(false);
        let cell_type = if is_dmz { BilliardCellType::Dmz } else { BilliardCellType::NonDmz };

        // Count DMZ transitions
        if let Some(last) = symbolic.last() {
            if *last != cell_type {
                n_dmz_transitions += 1;
            }
        }

        symbolic.push(cell_type);
        visited.insert((row, col));
    }

    // Compute entropy rate from bigram distribution
    let entropy_rate = compute_bigram_entropy(&symbolic);
    let actual_steps = symbolic.len();
    let mean_free_path = if n_reflections > 0 {
        actual_steps as f64 / n_reflections as f64
    } else {
        actual_steps as f64
    };
    let dmz_transition_rate = if actual_steps > 1 {
        n_dmz_transitions as f64 / (actual_steps - 1) as f64
    } else {
        0.0
    };

    EtBilliardTrajectory {
        n, s, k,
        n_steps: actual_steps,
        symbolic_dynamics: symbolic,
        n_distinct_cells: visited.len(),
        coverage: visited.len() as f64 / n_valid as f64,
        entropy_rate,
        n_reflections,
        mean_free_path,
        dmz_transition_rate,
        n_valid_cells: n_valid,
        n_dmz_cells: n_dmz,
        fill_ratio: if n_valid > 0 { n_dmz as f64 / n_valid as f64 } else { 0.0 },
    }
}

/// Compute Shannon entropy rate of bigram distribution in a symbolic sequence.
///
/// H_2 = -sum_{(a,b)} p(a,b) * log(p(a,b) / p(a))
/// where p(a,b) = count(a->b) / total_transitions and p(a) = count(a) / total.
fn compute_bigram_entropy(sequence: &[BilliardCellType]) -> f64 {
    if sequence.len() < 2 {
        return 0.0;
    }

    // Count bigrams: (from_type, to_type) -> count
    // 2 cell types: Dmz=0, NonDmz=1
    let type_idx = |t: &BilliardCellType| -> usize {
        match t {
            BilliardCellType::Dmz => 0,
            BilliardCellType::NonDmz => 1,
        }
    };

    let mut bigram = [[0u64; 2]; 2];
    let mut unigram = [0u64; 2];

    for pair in sequence.windows(2) {
        let a = type_idx(&pair[0]);
        let b = type_idx(&pair[1]);
        bigram[a][b] += 1;
        unigram[a] += 1;
    }

    let total = unigram.iter().sum::<u64>() as f64;
    if total < 1.0 {
        return 0.0;
    }

    let mut h = 0.0;
    for (row, &u) in bigram.iter().zip(unigram.iter()) {
        let p_a = u as f64 / total;
        if p_a < 1e-15 {
            continue;
        }
        for &count in row {
            let p_ab = count as f64 / total;
            if p_ab > 1e-15 {
                // Conditional entropy contribution: -p(a,b) * log(p(b|a))
                // where p(b|a) = p(a,b) / p(a)
                h -= p_ab * (p_ab / p_a).ln();
            }
        }
    }
    h
}

/// Result of a phase sweep: billiard dynamics as a function of strut constant S.
#[derive(Debug, Clone)]
pub struct EtBilliardPhaseSweep {
    /// CD level N.
    pub n: usize,
    /// Individual trajectory results per S.
    pub trajectories: Vec<EtBilliardTrajectory>,
    /// Correlation between fill_ratio and entropy_rate across S values.
    /// Negative correlation expected: high fill -> low entropy (ordered).
    pub fill_entropy_correlation: f64,
    /// Correlation between fill_ratio and mean_free_path.
    pub fill_mfp_correlation: f64,
}

/// Sweep the ET billiard across all valid strut constants for a given CD level.
///
/// For each S in 1..G (where G = 2^{N-1}), simulate a billiard trajectory
/// and record dynamical metrics. Returns aggregate statistics including
/// correlation between fill ratio and dynamical measures.
pub fn et_billiard_phase_sweep(
    n: usize,
    n_steps: usize,
    seed: u64,
) -> EtBilliardPhaseSweep {
    assert!(n >= 4, "Need at least sedenions (N >= 4)");
    let g = 1usize << (n - 1);
    let mut trajectories = Vec::with_capacity(g - 1);

    for s in 1..g {
        let traj = simulate_et_billiard(n, s, n_steps, seed.wrapping_add(s as u64));
        trajectories.push(traj);
    }

    // Compute correlations
    let fill_entropy_correlation = pearson_correlation(
        &trajectories.iter().map(|t| t.fill_ratio).collect::<Vec<_>>(),
        &trajectories.iter().map(|t| t.entropy_rate).collect::<Vec<_>>(),
    );
    let fill_mfp_correlation = pearson_correlation(
        &trajectories.iter().map(|t| t.fill_ratio).collect::<Vec<_>>(),
        &trajectories.iter().map(|t| t.mean_free_path).collect::<Vec<_>>(),
    );

    EtBilliardPhaseSweep {
        n,
        trajectories,
        fill_entropy_correlation,
        fill_mfp_correlation,
    }
}

/// Pearson correlation coefficient between two equal-length f64 slices.
/// Returns 0.0 if insufficient data or zero variance.
fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n < 2 {
        return 0.0;
    }

    let mean_x: f64 = x.iter().sum::<f64>() / n as f64;
    let mean_y: f64 = y.iter().sum::<f64>() / n as f64;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for i in 0..n {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let denom = (var_x * var_y).sqrt();
    if denom < 1e-15 {
        return 0.0;
    }
    cov / denom
}

/// Compare billiard dynamics against spectroscopy band classification.
///
/// For each spectroscopy band, compute the mean entropy rate and mean free path
/// of billiard trajectories within that band's strut range. Returns a comparison
/// table that tests whether band classification predicts dynamical phase.
#[derive(Debug, Clone)]
pub struct BandDynamicsComparison {
    /// CD level N.
    pub n: usize,
    /// Per-band comparison entries.
    pub entries: Vec<BandDynamicsEntry>,
}

/// One entry in the band-dynamics comparison.
#[derive(Debug, Clone)]
pub struct BandDynamicsEntry {
    /// Band index.
    pub band_index: usize,
    /// Band behavior classification from spectroscopy.
    pub behavior: String,
    /// Strut range [s_lo, s_hi].
    pub s_lo: usize,
    pub s_hi: usize,
    /// Mean entropy rate across struts in this band.
    pub mean_entropy_rate: f64,
    /// Std dev of entropy rate.
    pub std_entropy_rate: f64,
    /// Mean free path across struts in this band.
    pub mean_free_path: f64,
    /// Mean DMZ transition rate.
    pub mean_dmz_transition_rate: f64,
    /// Mean cell coverage.
    pub mean_coverage: f64,
    /// Mean fill ratio.
    pub mean_fill_ratio: f64,
}

/// Run the complete Experiment B: billiard dynamics vs spectroscopy bands.
///
/// 1. Compute spectroscopy bands for CD level N.
/// 2. Simulate billiard trajectories for each strut constant S.
/// 3. Group trajectories by spectroscopy band.
/// 4. Compare mean dynamical metrics across band types.
pub fn experiment_b_billiard_vs_spectroscopy(
    n: usize,
    n_steps: usize,
    seed: u64,
) -> BandDynamicsComparison {
    use super::emanation::spectroscopy_bands;

    let spec = spectroscopy_bands(n);
    let sweep = et_billiard_phase_sweep(n, n_steps, seed);

    let mut entries = Vec::with_capacity(spec.bands.len());

    for band in &spec.bands {
        // Collect trajectories whose S falls in this band
        let band_trajs: Vec<&EtBilliardTrajectory> = sweep.trajectories.iter()
            .filter(|t| t.s >= band.s_lo && t.s <= band.s_hi)
            .collect();

        if band_trajs.is_empty() {
            continue;
        }

        let n_t = band_trajs.len() as f64;
        let mean_entropy = band_trajs.iter().map(|t| t.entropy_rate).sum::<f64>() / n_t;
        let std_entropy = if band_trajs.len() > 1 {
            let var = band_trajs.iter()
                .map(|t| (t.entropy_rate - mean_entropy).powi(2))
                .sum::<f64>() / (n_t - 1.0);
            var.sqrt()
        } else {
            0.0
        };
        let mean_mfp = band_trajs.iter().map(|t| t.mean_free_path).sum::<f64>() / n_t;
        let mean_dmz_tr = band_trajs.iter().map(|t| t.dmz_transition_rate).sum::<f64>() / n_t;
        let mean_cov = band_trajs.iter().map(|t| t.coverage).sum::<f64>() / n_t;
        let mean_fill = band_trajs.iter().map(|t| t.fill_ratio).sum::<f64>() / n_t;

        let behavior_str = format!("{:?}", band.behavior);

        entries.push(BandDynamicsEntry {
            band_index: band.band_index,
            behavior: behavior_str,
            s_lo: band.s_lo,
            s_hi: band.s_hi,
            mean_entropy_rate: mean_entropy,
            std_entropy_rate: std_entropy,
            mean_free_path: mean_mfp,
            mean_dmz_transition_rate: mean_dmz_tr,
            mean_coverage: mean_cov,
            mean_fill_ratio: mean_fill,
        });
    }

    BandDynamicsComparison { n, entries }
}

// =====================================================================
// Sky-Limit-Set Correspondence (Experiment C)
// =====================================================================

/// Skybox pattern invariants extracted from the Emanation Table.
#[derive(Debug, Clone)]
pub struct SkyboxInvariants {
    /// CD level N.
    pub n: usize,
    /// Strut constant S.
    pub s: usize,
    /// Skybox edge length G = 2^{N-1}.
    pub edge: usize,
    /// DMZ density: dmz_count / (edge * edge).
    pub dmz_density: f64,
    /// Number of DMZ cells.
    pub dmz_count: usize,
    /// Number of structural empty cells (diagonal + anti-diagonal).
    pub n_structural_empty: usize,
    /// Number of connected components of empty (non-DMZ, non-structural) cells.
    pub n_empty_components: usize,
    /// Size of the largest connected component of empty cells.
    pub largest_empty_component: usize,
    /// Label-line DMZ count.
    pub label_dmz_count: usize,
    /// Interior DMZ density (excluding label lines and structural empties).
    pub interior_dmz_density: f64,
}

/// Extract skybox pattern invariants for a given (N, S).
pub fn extract_skybox_invariants(n: usize, s: usize) -> SkyboxInvariants {
    use super::emanation::create_skybox;

    let sb = create_skybox(n, s);
    let edge = sb.edge;
    let total = edge * edge;

    // Count structural empties and build empty-cell map for component analysis
    let mut n_structural = 0usize;
    let mut is_empty = vec![vec![false; edge]; edge];
    let mut n_interior = 0usize;
    let mut n_interior_dmz = 0usize;

    for (grid_row, empty_row) in sb.grid.iter().zip(is_empty.iter_mut()) {
        for (cell, is_emp) in grid_row.iter().zip(empty_row.iter_mut()) {
            if cell.is_structural_empty {
                n_structural += 1;
            }
            // Empty = not DMZ and not structural empty (navigable non-DMZ space)
            if !cell.is_dmz && !cell.is_structural_empty {
                *is_emp = true;
            }
            // Interior cells: not label line, not structural empty
            if !cell.is_label_line && !cell.is_structural_empty {
                n_interior += 1;
                if cell.is_dmz {
                    n_interior_dmz += 1;
                }
            }
        }
    }

    // BFS to find connected components of empty cells (4-connected).
    // Uses index-based loops because BFS requires random access to visited[nr][nc]
    // for arbitrary neighbor coordinates.
    let mut visited = vec![vec![false; edge]; edge];
    let mut n_components = 0usize;
    let mut largest_component = 0usize;

    #[allow(clippy::needless_range_loop)]
    for r in 0..edge {
        for c in 0..edge {
            if is_empty[r][c] && !visited[r][c] {
                // BFS from (r, c)
                let mut queue = std::collections::VecDeque::new();
                queue.push_back((r, c));
                visited[r][c] = true;
                let mut comp_size = 0usize;

                while let Some((cr, cc)) = queue.pop_front() {
                    comp_size += 1;
                    for &(dr, dc) in &[(-1i32, 0), (1, 0), (0, -1i32), (0, 1)] {
                        let nr = cr as i32 + dr;
                        let nc = cc as i32 + dc;
                        if nr >= 0 && nr < edge as i32 && nc >= 0 && nc < edge as i32 {
                            let nr = nr as usize;
                            let nc = nc as usize;
                            if is_empty[nr][nc] && !visited[nr][nc] {
                                visited[nr][nc] = true;
                                queue.push_back((nr, nc));
                            }
                        }
                    }
                }

                n_components += 1;
                if comp_size > largest_component {
                    largest_component = comp_size;
                }
            }
        }
    }

    let interior_dmz_density = if n_interior > 0 {
        n_interior_dmz as f64 / n_interior as f64
    } else {
        0.0
    };

    SkyboxInvariants {
        n, s, edge,
        dmz_density: sb.dmz_count as f64 / total as f64,
        dmz_count: sb.dmz_count,
        n_structural_empty: n_structural,
        n_empty_components: n_components,
        largest_empty_component: largest_component,
        label_dmz_count: sb.label_dmz_count,
        interior_dmz_density,
    }
}

/// Coxeter group type for comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoxeterType {
    /// A_n: symmetric group S_{n+1}
    A,
    /// B_n: hyperoctahedral group
    B,
    /// D_n: even-signed permutations
    D,
}

/// Invariants of a Coxeter group computed from its Coxeter matrix.
#[derive(Debug, Clone)]
pub struct CoxeterInvariants {
    /// Group type.
    pub group_type: CoxeterType,
    /// Rank n.
    pub rank: usize,
    /// Coxeter number h (order of Coxeter element).
    pub coxeter_number: usize,
    /// Order of the Weyl group |W|.
    pub group_order: f64,
    /// Number of positive roots.
    pub n_positive_roots: usize,
    /// Spectral radius of the Coxeter matrix (largest eigenvalue magnitude).
    /// For finite groups this is 2*cos(pi/h).
    pub spectral_radius: f64,
    /// Determinant of the Cartan matrix.
    pub cartan_determinant: f64,
}

/// Compute Coxeter group invariants for A_n, B_n, or D_n.
///
/// These are computed from closed-form expressions (no matrix diagonalization needed).
pub fn coxeter_invariants(group_type: CoxeterType, rank: usize) -> CoxeterInvariants {
    assert!(rank >= 1, "Rank must be at least 1");

    match group_type {
        CoxeterType::A => {
            let h = rank + 1; // Coxeter number for A_n
            let order = factorial(rank + 1); // |S_{n+1}| = (n+1)!
            let n_pos = rank * (rank + 1) / 2;
            let spectral_radius = 2.0 * (std::f64::consts::PI / h as f64).cos();
            let det = (rank + 1) as f64; // det(Cartan A_n) = n+1
            CoxeterInvariants {
                group_type, rank, coxeter_number: h,
                group_order: order,
                n_positive_roots: n_pos,
                spectral_radius,
                cartan_determinant: det,
            }
        }
        CoxeterType::B => {
            let h = 2 * rank; // Coxeter number for B_n
            let order = (2.0_f64).powi(rank as i32) * factorial(rank); // 2^n * n!
            let n_pos = rank * rank;
            let spectral_radius = 2.0 * (std::f64::consts::PI / h as f64).cos();
            let det = 2.0; // det(Cartan B_n) = 2
            CoxeterInvariants {
                group_type, rank, coxeter_number: h,
                group_order: order,
                n_positive_roots: n_pos,
                spectral_radius,
                cartan_determinant: det,
            }
        }
        CoxeterType::D => {
            assert!(rank >= 2, "D_n requires rank >= 2");
            let h = 2 * (rank - 1); // Coxeter number for D_n
            let order = (2.0_f64).powi(rank as i32 - 1) * factorial(rank); // 2^{n-1} * n!
            let n_pos = rank * (rank - 1);
            let spectral_radius = if h > 0 {
                2.0 * (std::f64::consts::PI / h as f64).cos()
            } else {
                0.0
            };
            let det = 4.0; // det(Cartan D_n) = 4
            CoxeterInvariants {
                group_type, rank, coxeter_number: h,
                group_order: order,
                n_positive_roots: n_pos,
                spectral_radius,
                cartan_determinant: det,
            }
        }
    }
}

/// Factorial as f64 (for group order computation).
fn factorial(n: usize) -> f64 {
    (1..=n).map(|i| i as f64).product()
}

/// Comparison result between skybox pattern and Coxeter group invariants.
#[derive(Debug, Clone)]
pub struct SkyLimitSetComparison {
    /// CD level N.
    pub n: usize,
    /// Skybox invariants aggregated across all valid strut constants.
    pub mean_dmz_density: f64,
    pub mean_n_empty_components: f64,
    pub mean_interior_dmz_density: f64,
    /// Box-kite count for this level (= N - 1 for sedenions and above).
    pub box_kite_count: usize,
    /// Coxeter group comparisons: one per candidate group.
    pub coxeter_comparisons: Vec<CoxeterMatchEntry>,
}

/// Match quality between skybox invariants and a Coxeter group.
#[derive(Debug, Clone)]
pub struct CoxeterMatchEntry {
    /// Coxeter group invariants.
    pub coxeter: CoxeterInvariants,
    /// Rank ratio: box_kite_count / coxeter_rank. Ideal = 1.0.
    pub rank_ratio: f64,
    /// DMZ density vs normalized root count: dmz_density / (n_positive_roots / rank^2).
    pub density_root_ratio: f64,
    /// Spectral radius of Coxeter matrix.
    pub spectral_radius: f64,
    /// Composite match score (heuristic, lower = better match).
    pub match_score: f64,
}

/// Run the complete Experiment C: compare ET skybox invariants to Coxeter limit set invariants.
///
/// For each CD level N in the provided list:
/// 1. Compute skybox invariants across all strut constants
/// 2. Compute Coxeter invariants for A_{N-1}, B_{N-1}, D_{N-1}
/// 3. Compare via rank matching and structural correspondence
///
/// Returns a comparison table per level.
pub fn experiment_c_sky_limit_set(
    levels: &[usize],
) -> Vec<SkyLimitSetComparison> {
    let mut results = Vec::new();

    for &n in levels {
        assert!(n >= 4, "Need at least sedenions (N >= 4)");
        let g = 1usize << (n - 1);

        // Collect skybox invariants across all strut constants
        let mut dmz_densities = Vec::new();
        let mut n_empty_comps = Vec::new();
        let mut interior_densities = Vec::new();

        for s in 1..g {
            let inv = extract_skybox_invariants(n, s);
            dmz_densities.push(inv.dmz_density);
            n_empty_comps.push(inv.n_empty_components as f64);
            interior_densities.push(inv.interior_dmz_density);
        }

        let mean_dmz = dmz_densities.iter().sum::<f64>() / dmz_densities.len() as f64;
        let mean_empty = n_empty_comps.iter().sum::<f64>() / n_empty_comps.len() as f64;
        let mean_interior = interior_densities.iter().sum::<f64>() / interior_densities.len() as f64;
        let bk_count = if n >= 4 { n - 1 } else { 0 };

        // Compare against Coxeter groups at matching rank
        let candidate_rank = bk_count; // try rank = box-kite count
        let mut comparisons = Vec::new();

        for &group_type in &[CoxeterType::A, CoxeterType::B, CoxeterType::D] {
            if candidate_rank < 2 && group_type == CoxeterType::D {
                continue; // D_n needs rank >= 2
            }
            let cox = coxeter_invariants(group_type, candidate_rank.max(1));
            let rank_ratio = bk_count as f64 / cox.rank as f64;

            // Normalized root density: n_positive_roots / rank^2
            let norm_root_density = cox.n_positive_roots as f64 / (cox.rank * cox.rank) as f64;
            let density_root_ratio = if norm_root_density > 0.0 {
                mean_dmz / norm_root_density
            } else {
                0.0
            };

            // Match score: |rank_ratio - 1| + |density_root_ratio - 1|
            // Lower = better correspondence
            let match_score = (rank_ratio - 1.0).abs() + (density_root_ratio - 1.0).abs();

            comparisons.push(CoxeterMatchEntry {
                coxeter: cox,
                rank_ratio,
                density_root_ratio,
                spectral_radius: coxeter_invariants(group_type, candidate_rank.max(1)).spectral_radius,
                match_score,
            });
        }

        // Sort by match score (best match first)
        comparisons.sort_by(|a, b| a.match_score.partial_cmp(&b.match_score).unwrap_or(std::cmp::Ordering::Equal));

        results.push(SkyLimitSetComparison {
            n,
            mean_dmz_density: mean_dmz,
            mean_n_empty_components: mean_empty,
            mean_interior_dmz_density: mean_interior,
            box_kite_count: bk_count,
            coxeter_comparisons: comparisons,
        });
    }

    results
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
    fn test_permutation_test_highly_local() {
        let sys = E10DynkinSystem;
        // A sequence with many adjacent pairs should have low p-value
        let seq = vec![0, 1, 2, 3, 4, 5, 4, 6, 7, 6, 4, 3, 2, 1, 0, 8, 9];
        let p = permutation_test_generic(&sys, &seq, 100, 42);
        assert!(p < 0.1, "Highly local sequence should have low p-value, got {}", p);
    }

    // === Walk generator tests ===

    #[test]
    fn test_neighbor_walk_length() {
        let sys = E10DynkinSystem;
        let seq = neighbor_walk(&sys, 100, 42);
        assert_eq!(seq.len(), 100);
        // All elements should be valid generator indices
        for &g in &seq {
            assert!(g < 10, "Generator {} out of range", g);
        }
    }

    #[test]
    fn test_neighbor_walk_is_local() {
        let sys = E10DynkinSystem;
        let seq = neighbor_walk(&sys, 1000, 42);
        let m = compute_generic_locality(&sys, &seq);
        // Neighbor walk should produce high locality (close to 1.0)
        assert!(m.r > 0.8, "Neighbor walk should be highly local, got r={}", m.r);
    }

    #[test]
    fn test_uniform_random_near_null() {
        let sys = E10DynkinSystem;
        let seq = uniform_random_sequence(&sys, 5000, 42);
        let m = compute_generic_locality(&sys, &seq);
        // Uniform random should be close to r_null = 0.2
        assert!(
            (m.r - m.r_null).abs() < 0.05,
            "Uniform random r={:.4} should be near r_null={:.4}", m.r, m.r_null
        );
    }

    #[test]
    fn test_neighbor_walk_on_et() {
        let sys = EtDmzSystem::new(5, 9);
        let seq = neighbor_walk(&sys, 200, 42);
        assert_eq!(seq.len(), 200);
        let m = compute_generic_locality(&sys, &seq);
        // Should have some locality (depends on ET structure)
        assert!(m.n_transitions > 0);
    }

    #[test]
    fn test_neighbor_walk_on_sedenion_zd() {
        let sys = SedenionZdSystem::new();
        let seq = neighbor_walk(&sys, 500, 42);
        assert_eq!(seq.len(), 500);
        let m = compute_generic_locality(&sys, &seq);
        // Should be highly local on a dense graph
        assert!(m.r > 0.5, "ZD neighbor walk r={:.4} should be substantial", m.r);
    }

    #[test]
    fn test_cross_stack_comparison() {
        let e10 = E10DynkinSystem;
        let zd = SedenionZdSystem::new();
        let et = EtDmzSystem::new(5, 9);
        let systems: Vec<&dyn ConstraintSystem> = vec![&e10, &zd, &et];

        let result = cross_stack_comparison(&systems, 500, 50, 42);

        assert_eq!(result.walk_metrics.len(), 3);
        assert_eq!(result.null_metrics.len(), 3);
        assert_eq!(result.p_values.len(), 3);

        // All walk metrics should show higher r than null
        for (wm, nm) in result.walk_metrics.iter().zip(result.null_metrics.iter()) {
            assert!(
                wm.r > nm.r,
                "{}: walk r={:.4} should exceed null r={:.4}",
                wm.system_name, wm.r, nm.r
            );
        }
    }

    #[test]
    fn test_walk_deterministic_with_seed() {
        let sys = E10DynkinSystem;
        let seq1 = neighbor_walk(&sys, 100, 42);
        let seq2 = neighbor_walk(&sys, 100, 42);
        assert_eq!(seq1, seq2, "Same seed should produce same walk");

        let seq3 = neighbor_walk(&sys, 100, 99);
        assert_ne!(seq1, seq3, "Different seeds should produce different walks");
    }

    #[test]
    fn test_walk_no_self_transitions() {
        let sys = E10DynkinSystem;
        let seq = random_walk_on_graph(&sys, 1000, 0.5, 42);
        for pair in seq.windows(2) {
            assert_ne!(pair[0], pair[1], "No self-transitions allowed");
        }
    }

    // === Twist Navigation System tests ===

    #[test]
    fn test_twist_system_basic() {
        let sys = TwistNavigationSystem::new();
        assert_eq!(sys.n_generators(), 7);
        assert_eq!(sys.name(), "Twist navigation (Twisted Sisters)");
        // All 7 struts should be mapped
        for &s in &[1, 3, 5, 7, 9, 11, 13] {
            assert!(sys.strut_index(s).is_some(), "Strut {} should be indexed", s);
        }
        // Even numbers should not be mapped
        assert!(sys.strut_index(2).is_none());
        assert!(sys.strut_index(0).is_none());
    }

    #[test]
    fn test_twist_system_has_edges() {
        let sys = TwistNavigationSystem::new();
        let mut edge_count = 0;
        for i in 0..7 {
            for j in (i + 1)..7 {
                if sys.adjacent(i, j) {
                    edge_count += 1;
                }
            }
        }
        // Twisted Sisters graph should have edges (we know it connects all 7)
        assert!(edge_count > 0, "Twist graph should have edges");
        assert!(edge_count <= 21, "At most C(7,2)=21 edges");
    }

    #[test]
    fn test_twist_walk_locality() {
        let sys = TwistNavigationSystem::new();
        let walk = neighbor_walk(&sys, 500, 42);
        let m = compute_generic_locality(&sys, &walk);
        // Neighbor walk on a connected graph should be highly local
        assert!(m.r > 0.5, "Twist walk r={:.4} should be substantial", m.r);
    }

    #[test]
    fn test_four_system_cross_stack() {
        // The complete Experiment A comparison: 4 constraint systems
        let e10 = E10DynkinSystem;
        let zd = SedenionZdSystem::new();
        let et = EtDmzSystem::new(5, 9);
        let twist = TwistNavigationSystem::new();
        let systems: Vec<&dyn ConstraintSystem> = vec![&e10, &zd, &et, &twist];

        let result = cross_stack_comparison(&systems, 300, 30, 42);
        assert_eq!(result.walk_metrics.len(), 4);

        // All should show locality greater than null
        for (wm, nm) in result.walk_metrics.iter().zip(result.null_metrics.iter()) {
            assert!(
                wm.r > nm.r,
                "{}: walk r={:.4} should exceed null r={:.4}",
                wm.system_name, wm.r, nm.r
            );
        }
    }

    // === ET Discrete Billiard tests (Experiment B) ===

    #[test]
    fn test_billiard_direction_rotation() {
        assert_eq!(BilliardDirection::Up.rotate_cw(), BilliardDirection::Right);
        assert_eq!(BilliardDirection::Right.rotate_cw(), BilliardDirection::Down);
        assert_eq!(BilliardDirection::Down.rotate_cw(), BilliardDirection::Left);
        assert_eq!(BilliardDirection::Left.rotate_cw(), BilliardDirection::Up);
    }

    #[test]
    fn test_billiard_direction_reverse() {
        assert_eq!(BilliardDirection::Up.reverse(), BilliardDirection::Down);
        assert_eq!(BilliardDirection::Left.reverse(), BilliardDirection::Right);
    }

    #[test]
    fn test_billiard_basic_trajectory() {
        // N=4 (sedenions), S=1: should have a valid ET grid
        let traj = simulate_et_billiard(4, 1, 100, 42);
        assert_eq!(traj.n, 4);
        assert_eq!(traj.s, 1);
        assert_eq!(traj.n_steps, 100);
        assert_eq!(traj.symbolic_dynamics.len(), 100);
        // Should visit at least 1 cell
        assert!(traj.n_distinct_cells >= 1);
        // Coverage should be between 0 and 1
        assert!(traj.coverage >= 0.0 && traj.coverage <= 1.0);
        // Entropy rate should be non-negative
        assert!(traj.entropy_rate >= 0.0);
    }

    #[test]
    fn test_billiard_n5_produces_valid_trajectory() {
        // N=5, S=9: standard test case from Engine C
        let traj = simulate_et_billiard(5, 9, 500, 42);
        assert_eq!(traj.n, 5);
        assert_eq!(traj.s, 9);
        assert!(traj.k > 0);
        assert!(traj.n_valid_cells > 0);
        assert!(traj.n_distinct_cells > 0);
        // Mean free path should be positive
        assert!(traj.mean_free_path > 0.0);
    }

    #[test]
    fn test_billiard_deterministic_with_seed() {
        let traj1 = simulate_et_billiard(4, 3, 200, 42);
        let traj2 = simulate_et_billiard(4, 3, 200, 42);
        assert_eq!(traj1.symbolic_dynamics, traj2.symbolic_dynamics);
        assert_eq!(traj1.n_reflections, traj2.n_reflections);

        let traj3 = simulate_et_billiard(4, 3, 200, 99);
        // Different seed should produce different trajectory (very likely)
        assert_ne!(traj1.n_reflections, traj3.n_reflections);
    }

    #[test]
    fn test_billiard_full_fill_low_entropy() {
        // S=1 at N=4 (Generator class) should be full-fill => low DMZ transition rate
        let traj = simulate_et_billiard(4, 1, 500, 42);
        // In a full-fill grid, all cells are DMZ, so DMZ transition rate should be 0
        if traj.fill_ratio > 0.99 {
            assert!(
                traj.dmz_transition_rate < 0.01,
                "Full-fill should have near-zero DMZ transition rate, got {}",
                traj.dmz_transition_rate
            );
        }
    }

    #[test]
    fn test_billiard_phase_sweep_n4() {
        // N=4: G=8, so S in 1..7 (7 strut constants)
        let sweep = et_billiard_phase_sweep(4, 200, 42);
        assert_eq!(sweep.n, 4);
        assert_eq!(sweep.trajectories.len(), 7);
        // Each trajectory should be valid
        for traj in &sweep.trajectories {
            assert_eq!(traj.n_steps, 200);
            assert!(traj.n_valid_cells > 0);
        }
    }

    #[test]
    fn test_bigram_entropy_constant_sequence() {
        // All-DMZ sequence: entropy should be 0 (only one bigram type)
        let seq: Vec<BilliardCellType> = vec![BilliardCellType::Dmz; 100];
        let h = compute_bigram_entropy(&seq);
        assert!(h.abs() < 1e-10, "Constant sequence entropy should be 0, got {}", h);
    }

    #[test]
    fn test_bigram_entropy_alternating_sequence() {
        // Alternating DMZ/NonDmz: high entropy (both bigram types equally likely)
        let mut seq = Vec::new();
        for i in 0..200 {
            seq.push(if i % 2 == 0 { BilliardCellType::Dmz } else { BilliardCellType::NonDmz });
        }
        let h = compute_bigram_entropy(&seq);
        // Alternating: bigrams are always Dmz->NonDmz or NonDmz->Dmz
        // H(X_{t+1}|X_t) = 0 because the next symbol is deterministic given current
        assert!(h.abs() < 1e-10, "Alternating sequence has 0 conditional entropy, got {}", h);
    }

    #[test]
    fn test_bigram_entropy_random_mix() {
        // Random-ish mix: should have positive entropy
        use BilliardCellType::*;
        let seq = vec![Dmz, Dmz, NonDmz, Dmz, NonDmz, NonDmz, Dmz, NonDmz, Dmz, Dmz,
                       NonDmz, NonDmz, Dmz, Dmz, Dmz, NonDmz, NonDmz, Dmz, NonDmz, Dmz];
        let h = compute_bigram_entropy(&seq);
        assert!(h > 0.0, "Mixed sequence should have positive entropy, got {}", h);
    }

    #[test]
    fn test_experiment_b_basic() {
        // Run full Experiment B at N=4 with short trajectories
        let result = experiment_b_billiard_vs_spectroscopy(4, 200, 42);
        assert_eq!(result.n, 4);
        assert!(!result.entries.is_empty(), "Should have at least one band entry");
        // Check that all entries have valid metrics
        for entry in &result.entries {
            assert!(entry.mean_entropy_rate >= 0.0);
            assert!(entry.mean_free_path > 0.0);
            assert!(entry.mean_coverage >= 0.0 && entry.mean_coverage <= 1.0);
        }
    }

    #[test]
    fn test_pearson_correlation_perfect() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-10, "Perfect positive correlation, got {}", r);

        let y_neg = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        let r_neg = pearson_correlation(&x, &y_neg);
        assert!((r_neg + 1.0).abs() < 1e-10, "Perfect negative correlation, got {}", r_neg);
    }

    #[test]
    fn test_pearson_correlation_zero_variance() {
        let x = vec![5.0, 5.0, 5.0];
        let y = vec![1.0, 2.0, 3.0];
        let r = pearson_correlation(&x, &y);
        assert!(r.abs() < 1e-10, "Zero variance should give 0 correlation, got {}", r);
    }

    // === Sky-Limit-Set Correspondence tests (Experiment C) ===

    #[test]
    fn test_skybox_invariants_n4() {
        let inv = extract_skybox_invariants(4, 3);
        assert_eq!(inv.n, 4);
        assert_eq!(inv.s, 3);
        assert_eq!(inv.edge, 8); // G = 2^3 = 8
        // DMZ density should be between 0 and 1
        assert!(inv.dmz_density >= 0.0 && inv.dmz_density <= 1.0);
        // Structural empties: diagonal + anti-diagonal = 2*8 - 2 (center counted once if edge even)
        assert!(inv.n_structural_empty > 0);
        // n_empty_components is valid (usize, always >= 0)
    }

    #[test]
    fn test_skybox_invariants_n5() {
        let inv = extract_skybox_invariants(5, 9);
        assert_eq!(inv.n, 5);
        assert_eq!(inv.edge, 16); // G = 2^4 = 16
        assert!(inv.dmz_density >= 0.0);
        assert!(inv.n_structural_empty > 0);
    }

    #[test]
    fn test_coxeter_invariants_a_n() {
        // A_3: symmetric group S_4, h=4, |W|=24, n_pos=6
        let a3 = coxeter_invariants(CoxeterType::A, 3);
        assert_eq!(a3.coxeter_number, 4);
        assert!((a3.group_order - 24.0).abs() < 1e-10);
        assert_eq!(a3.n_positive_roots, 6);
        assert!((a3.cartan_determinant - 4.0).abs() < 1e-10);
        // Spectral radius = 2*cos(pi/4) = sqrt(2)
        assert!((a3.spectral_radius - std::f64::consts::SQRT_2).abs() < 1e-10);
    }

    #[test]
    fn test_coxeter_invariants_b_n() {
        // B_3: hyperoctahedral, h=6, |W|=2^3*3!=48, n_pos=9
        let b3 = coxeter_invariants(CoxeterType::B, 3);
        assert_eq!(b3.coxeter_number, 6);
        assert!((b3.group_order - 48.0).abs() < 1e-10);
        assert_eq!(b3.n_positive_roots, 9);
        assert!((b3.cartan_determinant - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_coxeter_invariants_d_n() {
        // D_4: h=6, |W|=2^3*4!=192, n_pos=12
        let d4 = coxeter_invariants(CoxeterType::D, 4);
        assert_eq!(d4.coxeter_number, 6);
        assert!((d4.group_order - 192.0).abs() < 1e-10);
        assert_eq!(d4.n_positive_roots, 12);
        assert!((d4.cartan_determinant - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_factorial() {
        assert!((factorial(0) - 1.0).abs() < 1e-10);
        assert!((factorial(1) - 1.0).abs() < 1e-10);
        assert!((factorial(5) - 120.0).abs() < 1e-10);
        assert!((factorial(10) - 3628800.0).abs() < 1e-10);
    }

    #[test]
    fn test_experiment_c_n4() {
        let results = experiment_c_sky_limit_set(&[4]);
        assert_eq!(results.len(), 1);
        let r = &results[0];
        assert_eq!(r.n, 4);
        assert_eq!(r.box_kite_count, 3); // N-1 = 3
        // Should have 3 Coxeter comparisons (A_3, B_3, D_3)
        assert_eq!(r.coxeter_comparisons.len(), 3);
        // Best match should have a finite match score
        assert!(r.coxeter_comparisons[0].match_score.is_finite());
        // Mean DMZ density should be positive
        assert!(r.mean_dmz_density > 0.0);
    }

    #[test]
    fn test_experiment_c_rank_matching() {
        // At N=4: box-kite count = 3, so we compare against rank-3 groups
        let results = experiment_c_sky_limit_set(&[4]);
        let r = &results[0];
        // All comparisons should have rank_ratio = 1.0 (since we use bk_count as rank)
        for comp in &r.coxeter_comparisons {
            assert!((comp.rank_ratio - 1.0).abs() < 1e-10,
                "Rank ratio should be 1.0, got {}", comp.rank_ratio);
        }
    }
}
