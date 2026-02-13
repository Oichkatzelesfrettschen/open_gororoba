//! Vietoris-Rips Complex and Persistent Homology
//!
//! Clean-room implementation of Vietoris-Rips filtration and standard persistent
//! homology algorithm for E-027 topology analysis.
//!
//! # Mathematics
//!
//! **Vietoris-Rips Complex**: For point cloud X and threshold r:
//! - 0-simplices: all points
//! - 1-simplices: edges (i,j) where d(x_i, x_j) <= r
//! - k-simplices: (i_0,...,i_k) where all pairwise distances <= r
//!
//! **Persistent Homology**: Tracks birth and death of topological features
//! (connected components, loops, voids) as filtration parameter r increases.
//!
//! # References
//!
//! - Edelsbrunner & Harer (2010). Computational Topology: An Introduction.
//! - Zomorodian & Carlsson (2005). Computing Persistent Homology.

use std::collections::HashSet;

/// A k-simplex (ordered vertex list)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Simplex {
    /// Ordered vertex indices
    pub vertices: Vec<usize>,
    /// Dimension (k = len(vertices) - 1)
    pub dim: usize,
    /// Birth time (filtration value when simplex appears)
    pub birth: OrderedFloat,
}

/// Wrapper for f64 that implements Eq/Hash for simplicial complex keys
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct OrderedFloat(pub f64);

impl Eq for OrderedFloat {}

impl std::hash::Hash for OrderedFloat {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

impl Simplex {
    /// Create new simplex with given vertices (will be sorted)
    pub fn new(mut vertices: Vec<usize>, birth: f64) -> Self {
        vertices.sort_unstable();
        let dim = vertices.len() - 1;
        Self {
            vertices,
            dim,
            birth: OrderedFloat(birth),
        }
    }

    /// Get boundary of this simplex (all (k-1)-faces)
    ///
    /// For edge (0,1): boundary is {(0), (1)}
    /// For triangle (0,1,2): boundary is {(0,1), (0,2), (1,2)}
    pub fn boundary(&self) -> Vec<Simplex> {
        if self.dim == 0 {
            return Vec::new(); // 0-simplex has empty boundary
        }

        let mut faces = Vec::with_capacity(self.vertices.len());
        for i in 0..self.vertices.len() {
            let mut face_verts = self.vertices.clone();
            face_verts.remove(i);
            faces.push(Simplex::new(face_verts, 0.0)); // birth time irrelevant for lookup
        }
        faces
    }

}

/// Distance matrix wrapper
#[derive(Debug, Clone)]
pub struct DistanceMatrix {
    n: usize,
    data: Vec<f64>,
}

impl DistanceMatrix {
    /// Create from flat upper-triangular data
    pub fn new(n: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), n * (n - 1) / 2, "Invalid distance matrix size");
        Self { n, data }
    }

    /// Get number of points in the matrix
    pub fn size(&self) -> usize {
        self.n
    }

    /// Get distance between points i and j
    pub fn get(&self, i: usize, j: usize) -> f64 {
        if i == j {
            return 0.0;
        }
        let (i, j) = if i < j { (i, j) } else { (j, i) };
        let idx = i * self.n - i * (i + 1) / 2 + (j - i - 1);
        self.data[idx]
    }

    /// Compute from 3D point cloud (n x 3 array)
    pub fn from_points_3d(points: &[f64]) -> Self {
        let n = points.len() / 3;
        assert_eq!(points.len() % 3, 0, "Points must be (n x 3) array");

        let mut data = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                let dx = points[i * 3] - points[j * 3];
                let dy = points[i * 3 + 1] - points[j * 3 + 1];
                let dz = points[i * 3 + 2] - points[j * 3 + 2];
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                data.push(dist);
            }
        }
        Self::new(n, data)
    }
}

/// Vietoris-Rips complex at given filtration value
#[derive(Debug, Clone)]
pub struct VietorisRipsComplex {
    /// All simplices in the complex, organized by dimension
    pub simplices: Vec<Vec<Simplex>>, // simplices[k] = all k-simplices
    /// Maximum dimension
    pub max_dim: usize,
}

impl VietorisRipsComplex {
    /// Build Vietoris-Rips complex up to max_dim
    ///
    /// # Algorithm
    ///
    /// 1. Add all 0-simplices (vertices) at birth time 0
    /// 2. For k=1 to max_dim:
    ///    - Add k-simplex if all its (k-1)-faces are present
    ///    - Birth time = max distance among all vertex pairs
    pub fn build(dist: &DistanceMatrix, max_threshold: f64, max_dim: usize) -> Self {
        let n = dist.n;
        let mut simplices = vec![Vec::new(); max_dim + 1];

        // Add vertices
        for i in 0..n {
            simplices[0].push(Simplex::new(vec![i], 0.0));
        }

        // Add edges (1-simplices)
        for i in 0..n {
            for j in (i + 1)..n {
                let d = dist.get(i, j);
                if d <= max_threshold {
                    simplices[1].push(Simplex::new(vec![i, j], d));
                }
            }
        }

        // Add higher-dimensional simplices
        for k in 2..=max_dim {
            let prev_simplices = &simplices[k - 1];
            let mut candidates = HashSet::new();

            // Generate candidate k-simplices from (k-1)-simplices
            for s1 in prev_simplices {
                for s2 in prev_simplices {
                    if s1.vertices != s2.vertices {
                        let mut union: Vec<usize> = s1.vertices.iter()
                            .chain(s2.vertices.iter())
                            .copied()
                            .collect::<HashSet<_>>()
                            .into_iter()
                            .collect();
                        union.sort_unstable();

                        if union.len() == k + 1 {
                            candidates.insert(union);
                        }
                    }
                }
            }

            // Check each candidate k-simplex
            for candidate in candidates {
                // Compute birth time (max pairwise distance)
                let mut max_dist: f64 = 0.0;
                for i in 0..candidate.len() {
                    for j in (i + 1)..candidate.len() {
                        let d = dist.get(candidate[i], candidate[j]);
                        max_dist = max_dist.max(d);
                    }
                }

                if max_dist <= max_threshold {
                    simplices[k].push(Simplex::new(candidate, max_dist));
                }
            }
        }

        Self { simplices, max_dim }
    }

    /// Get total number of simplices
    pub fn size(&self) -> usize {
        self.simplices.iter().map(|v| v.len()).sum()
    }
}

/// Persistence pair (feature birth and death)
#[derive(Debug, Clone)]
pub struct PersistencePair {
    /// Dimension of feature (0=component, 1=loop, 2=void)
    pub dim: usize,
    /// Birth time (filtration value when feature appears)
    pub birth: f64,
    /// Death time (filtration value when feature disappears)
    pub death: f64,
}

impl PersistencePair {
    /// Persistence (lifetime) of feature
    pub fn persistence(&self) -> f64 {
        self.death - self.birth
    }
}

/// Boundary matrix column (sparse representation)
#[derive(Debug, Clone)]
struct BoundaryColumn {
    /// Row indices where column has non-zero entries (mod 2)
    indices: Vec<usize>,
}

impl BoundaryColumn {
    fn new() -> Self {
        Self { indices: Vec::new() }
    }

    fn from_indices(mut indices: Vec<usize>) -> Self {
        indices.sort_unstable();
        indices.dedup();
        Self { indices }
    }

    /// Add another column (XOR operation in Z/2Z)
    fn add_column(&mut self, other: &BoundaryColumn) {
        let mut result = Vec::new();
        let mut i = 0;
        let mut j = 0;

        while i < self.indices.len() && j < other.indices.len() {
            if self.indices[i] < other.indices[j] {
                result.push(self.indices[i]);
                i += 1;
            } else if self.indices[i] > other.indices[j] {
                result.push(other.indices[j]);
                j += 1;
            } else {
                // Equal indices cancel out (mod 2)
                i += 1;
                j += 1;
            }
        }

        result.extend_from_slice(&self.indices[i..]);
        result.extend_from_slice(&other.indices[j..]);
        self.indices = result;
    }

    /// Get lowest (maximum) index in column, or None if empty
    fn lowest_one(&self) -> Option<usize> {
        self.indices.last().copied()
    }

    fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
}

/// Build boundary matrix for all simplices
fn build_boundary_matrix(complex: &VietorisRipsComplex) -> Vec<BoundaryColumn> {
    let mut all_simplices: Vec<&Simplex> = Vec::new();
    for dim_simplices in &complex.simplices {
        all_simplices.extend(dim_simplices.iter());
    }

    // Sort by (birth time, dimension)
    all_simplices.sort_by(|a, b| {
        a.birth.partial_cmp(&b.birth)
            .unwrap()
            .then_with(|| a.dim.cmp(&b.dim))
    });

    // Build index map: vertices -> column index (ignore birth time for lookup)
    let mut vertex_to_idx = std::collections::HashMap::new();
    for (idx, simplex) in all_simplices.iter().enumerate() {
        vertex_to_idx.insert(simplex.vertices.clone(), idx);
    }

    // Build boundary columns
    let mut boundary_matrix = Vec::new();
    for simplex in &all_simplices {
        if simplex.dim == 0 {
            // 0-simplex has empty boundary
            boundary_matrix.push(BoundaryColumn::new());
        } else {
            // k-simplex boundary is sum of (k-1)-faces
            let faces = simplex.boundary();
            let mut indices = Vec::new();
            for face in faces {
                // Look up by vertices only, ignoring birth time
                if let Some(&idx) = vertex_to_idx.get(&face.vertices) {
                    indices.push(idx);
                }
            }
            boundary_matrix.push(BoundaryColumn::from_indices(indices));
        }
    }

    boundary_matrix
}

/// Standard persistent homology algorithm via matrix reduction
///
/// # Algorithm
///
/// 1. Sort simplices by birth time
/// 2. Build boundary matrix (columns = simplices, rows = simplices)
/// 3. Reduce matrix left-to-right using column operations
/// 4. Extract persistence pairs:
///    - If column j reduces to zero: j is a birth (creates homology class)
///    - If column j reduces to have lowest one at i: i dies at j's birth time
///
/// # Returns
///
/// Vector of persistence pairs (birth, death) for each homology feature
pub fn compute_persistent_homology(
    complex: &VietorisRipsComplex,
) -> Vec<PersistencePair> {
    // Collect all simplices and sort by birth time
    let mut all_simplices: Vec<&Simplex> = Vec::new();
    for dim_simplices in &complex.simplices {
        all_simplices.extend(dim_simplices.iter());
    }
    all_simplices.sort_by(|a, b| {
        a.birth.partial_cmp(&b.birth)
            .unwrap()
            .then_with(|| a.dim.cmp(&b.dim))
    });

    // Build boundary matrix
    let mut boundary_matrix = build_boundary_matrix(complex);
    let n = boundary_matrix.len();

    // Track lowest ones: low[j] = i means column j has lowest one at row i
    let mut low: Vec<Option<usize>> = vec![None; n];

    // Matrix reduction: process columns left to right
    for j in 0..n {
        while let Some(i) = boundary_matrix[j].lowest_one() {
            // Check if another column has same lowest one
            let mut found_match = false;
            for k in 0..j {
                if low[k] == Some(i) {
                    // Add column k to column j (eliminates lowest one)
                    let col_k = boundary_matrix[k].clone();
                    boundary_matrix[j].add_column(&col_k);
                    found_match = true;
                    break;
                }
            }
            if !found_match {
                break;
            }
        }
        low[j] = boundary_matrix[j].lowest_one();
    }

    // Extract persistence pairs
    let mut pairs = Vec::new();
    let mut paired = vec![false; n];

    for j in 0..n {
        if let Some(i) = low[j] {
            // Column j kills row i: feature born at i dies at j
            let birth_time = all_simplices[i].birth.0;
            let death_time = all_simplices[j].birth.0;
            let dim = all_simplices[i].dim; // Dimension of dying feature

            pairs.push(PersistencePair {
                dim,
                birth: birth_time,
                death: death_time,
            });

            paired[i] = true;
            paired[j] = true;
        }
    }

    // Unpaired simplices represent features that never die (infinite persistence)
    for (idx, simplex) in all_simplices.iter().enumerate() {
        if !paired[idx] && boundary_matrix[idx].is_empty() {
            pairs.push(PersistencePair {
                dim: simplex.dim,
                birth: simplex.birth.0,
                death: f64::INFINITY,
            });
        }
    }

    pairs
}

/// Compute Betti numbers from persistence pairs
///
/// Betti number b_k = number of k-dimensional features that persist to infinity
/// OR have very long persistence (> min_persistence threshold)
///
/// # Note
///
/// For accurate Betti numbers, typically count only features with infinite
/// persistence or very long relative to the filtration range.
pub fn compute_betti_numbers(
    pairs: &[PersistencePair],
    min_persistence: f64,
) -> Vec<usize> {
    let max_dim = pairs.iter().map(|p| p.dim).max().unwrap_or(0);
    let mut betti = vec![0; max_dim + 1];

    for pair in pairs {
        // Count features with infinite persistence or very long finite persistence
        if pair.death.is_infinite() || pair.persistence() > min_persistence {
            betti[pair.dim] += 1;
        }
    }

    betti
}

/// Compute Betti numbers at a specific filtration time
///
/// Count features that are alive at the given time threshold
pub fn compute_betti_numbers_at_time(
    pairs: &[PersistencePair],
    time: f64,
) -> Vec<usize> {
    let max_dim = pairs.iter().map(|p| p.dim).max().unwrap_or(0);
    let mut betti = vec![0; max_dim + 1];

    for pair in pairs {
        // Feature is alive if birth <= time < death
        if pair.birth <= time && (pair.death.is_infinite() || pair.death > time) {
            betti[pair.dim] += 1;
        }
    }

    betti
}

/// A persistence diagram: a multiset of (birth, death) pairs for a given dimension.
///
/// Provides distance metrics between diagrams for quantitative topological comparison.
/// Two key distances:
/// - Wasserstein (W_p): optimal transport cost between diagrams (aggregate measure)
/// - Bottleneck (W_inf): maximum matching cost (worst-case measure)
///
/// Points at infinity are excluded from distance computations (they represent
/// essential homology classes that never die).
#[derive(Debug, Clone)]
pub struct PersistenceDiagram {
    /// Dimension of the features in this diagram
    pub dim: usize,
    /// Finite persistence pairs as (birth, death) points
    pub points: Vec<(f64, f64)>,
}

impl PersistenceDiagram {
    /// Build diagram for a given dimension from persistence pairs.
    ///
    /// Filters to the specified dimension and excludes infinite-death features.
    pub fn from_pairs(pairs: &[PersistencePair], dim: usize) -> Self {
        let points: Vec<(f64, f64)> = pairs
            .iter()
            .filter(|p| p.dim == dim && p.death.is_finite())
            .map(|p| (p.birth, p.death))
            .collect();
        Self { dim, points }
    }

    /// Build separate diagrams for each dimension present.
    pub fn from_pairs_all(pairs: &[PersistencePair]) -> Vec<Self> {
        let max_dim = pairs.iter().map(|p| p.dim).max().unwrap_or(0);
        (0..=max_dim)
            .map(|d| Self::from_pairs(pairs, d))
            .collect()
    }

    /// Number of finite persistence points in this diagram.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Whether the diagram has no finite persistence points.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Total persistence: sum of (death - birth) for all finite features.
    pub fn total_persistence(&self) -> f64 {
        self.points.iter().map(|(b, d)| d - b).sum()
    }

    /// Maximum persistence among all finite features.
    pub fn max_persistence(&self) -> f64 {
        self.points
            .iter()
            .map(|(b, d)| d - b)
            .fold(0.0_f64, f64::max)
    }

    /// Truncate to the k most persistent features.
    ///
    /// Sorts by persistence (death - birth) descending and keeps only the top k.
    /// This dramatically speeds up Wasserstein/bottleneck distance computations
    /// which are O(n^3) in the number of persistence points.
    pub fn truncate_to_top_k(&mut self, k: usize) {
        if self.points.len() <= k {
            return;
        }
        self.points
            .sort_by(|a, b| {
                let pa = a.1 - a.0;
                let pb = b.1 - b.0;
                pb.partial_cmp(&pa).unwrap_or(std::cmp::Ordering::Equal)
            });
        self.points.truncate(k);
    }

    /// Persistence entropy (Shannon entropy of normalized persistence values).
    ///
    /// H = -sum_i (p_i / P) * ln(p_i / P)
    ///
    /// where p_i = death_i - birth_i and P = sum(p_i).
    /// Measures topological complexity: high entropy means many features
    /// of similar persistence; low entropy means dominated by few features.
    /// Returns 0.0 for empty diagrams.
    pub fn persistence_entropy(&self) -> f64 {
        if self.points.is_empty() {
            return 0.0;
        }
        let persistences: Vec<f64> = self
            .points
            .iter()
            .map(|(b, d)| d - b)
            .filter(|&p| p > 0.0)
            .collect();
        if persistences.is_empty() {
            return 0.0;
        }
        let total: f64 = persistences.iter().sum();
        if total <= 0.0 {
            return 0.0;
        }
        let mut entropy = 0.0;
        for &p in &persistences {
            let w = p / total;
            if w > 0.0 {
                entropy -= w * w.ln();
            }
        }
        entropy
    }

    /// Wasserstein distance (W_p) between two persistence diagrams.
    ///
    /// The p-Wasserstein distance is the p-th root of the minimum cost matching
    /// between the two diagrams, where:
    /// - Off-diagonal points can match to each other at cost d(a,b)^p
    /// - Off-diagonal points can match to the diagonal at cost (pers/2)^p
    ///   (projecting to the nearest point on the diagonal y=x)
    ///
    /// Uses the Hungarian algorithm for exact optimal matching.
    /// For p=2 (default): captures aggregate topological difference.
    /// For p=infinity: use bottleneck_distance() instead.
    pub fn wasserstein_distance(&self, other: &PersistenceDiagram, p: f64) -> f64 {
        if self.is_empty() && other.is_empty() {
            return 0.0;
        }

        let n = self.points.len();
        let m = other.points.len();
        let total = n + m;

        if total == 0 {
            return 0.0;
        }

        // Build cost matrix for the augmented matching problem.
        // Rows 0..n: points from self
        // Rows n..n+m: diagonal partners for other.points
        // Cols 0..m: points from other
        // Cols m..m+n: diagonal partners for self.points
        let mut cost = vec![vec![0.0_f64; total]; total];

        for (i, &(b1, d1)) in self.points.iter().enumerate() {
            for (j, &(b2, d2)) in other.points.iter().enumerate() {
                let db = (b1 - b2).abs();
                let dd = (d1 - d2).abs();
                cost[i][j] = db.max(dd).powf(p);
            }
            let pers_cost = ((d1 - b1) / 2.0).abs().powf(p);
            for slot in &mut cost[i][m..total] {
                *slot = pers_cost;
            }
        }

        for (idx, &(b2, d2)) in other.points.iter().enumerate() {
            let row = n + idx;
            let pers = ((d2 - b2) / 2.0).abs();
            let pers_p = pers.powf(p);
            for (j, &(b2j, d2j)) in other.points.iter().enumerate() {
                cost[row][j] = if j == idx {
                    pers_p
                } else {
                    let pers_j = ((d2j - b2j) / 2.0).abs().powf(p);
                    pers_p + pers_j
                };
            }
            // Diagonal to diagonal: zero cost (already initialized)
        }

        let assignment = hungarian_algorithm(&cost);
        let total_cost: f64 = assignment
            .iter()
            .enumerate()
            .map(|(i, &j)| cost[i][j])
            .sum();

        total_cost.powf(1.0 / p)
    }

    /// Bottleneck distance (W_infinity) between two persistence diagrams.
    ///
    /// The bottleneck distance is the minimum over all matchings of the
    /// maximum matching cost. Equivalent to W_p as p -> infinity.
    ///
    /// Uses the fact that bottleneck = min over matchings of max cost,
    /// computed via the Hungarian algorithm on a binary cost matrix at
    /// each candidate threshold (bisection search).
    pub fn bottleneck_distance(&self, other: &PersistenceDiagram) -> f64 {
        if self.is_empty() && other.is_empty() {
            return 0.0;
        }

        // Collect all candidate distances (between points and to diagonal)
        let mut candidates = Vec::new();
        for &(b1, d1) in &self.points {
            for &(b2, d2) in &other.points {
                let db = (b1 - b2).abs();
                let dd = (d1 - d2).abs();
                candidates.push(db.max(dd));
            }
            candidates.push((d1 - b1) / 2.0);
        }
        for &(b2, d2) in &other.points {
            candidates.push((d2 - b2) / 2.0);
        }

        candidates.sort_by(|a, b| a.partial_cmp(b).unwrap());
        candidates.dedup();

        if candidates.is_empty() {
            return 0.0;
        }

        // Binary search: find minimum threshold where a perfect matching exists
        let mut lo = 0;
        let mut hi = candidates.len() - 1;

        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let threshold = candidates[mid];

            // Build bipartite graph: edge exists if cost <= threshold
            if can_match_within_threshold(self, other, threshold) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }

        candidates[lo]
    }
}

/// Check if a perfect matching exists with all costs <= threshold.
///
/// Uses Hopcroft-Karp-style augmenting paths for bipartite matching.
fn can_match_within_threshold(
    dgm1: &PersistenceDiagram,
    dgm2: &PersistenceDiagram,
    threshold: f64,
) -> bool {
    let n = dgm1.points.len();
    let m = dgm2.points.len();
    let total = n + m;
    let eps = 1e-12;

    // Build adjacency: row i can match to col j if cost <= threshold
    let mut adj = vec![Vec::new(); total];

    for (i, &(b1, d1)) in dgm1.points.iter().enumerate() {
        for (j, &(b2, d2)) in dgm2.points.iter().enumerate() {
            let db = (b1 - b2).abs();
            let dd = (d1 - d2).abs();
            if db.max(dd) <= threshold + eps {
                adj[i].push(j);
            }
        }
        // Match to diagonal if persistence/2 <= threshold
        let pers = (d1 - b1) / 2.0;
        if pers <= threshold + eps {
            adj[i].extend(m..total);
        }
    }

    for (idx, &(b2, d2)) in dgm2.points.iter().enumerate() {
        let row = n + idx;
        let pers = (d2 - b2) / 2.0;
        if pers <= threshold + eps {
            adj[row].extend(m..total);
        }
        // Diagonal-to-diagonal always allowed
        for j in m..total {
            if !adj[row].contains(&j) {
                adj[row].push(j);
            }
        }
    }

    // Find maximum matching via augmenting paths
    let mut match_col: Vec<Option<usize>> = vec![None; total];
    let mut matched = 0;

    for i in 0..total {
        let mut visited = vec![false; total];
        if augment(i, &adj, &mut match_col, &mut visited) {
            matched += 1;
        }
    }

    matched == total
}

/// DFS augmenting path for bipartite matching.
fn augment(
    u: usize,
    adj: &[Vec<usize>],
    match_col: &mut [Option<usize>],
    visited: &mut [bool],
) -> bool {
    for &v in &adj[u] {
        if !visited[v] {
            visited[v] = true;
            if match_col[v].is_none()
                || augment(match_col[v].unwrap(), adj, match_col, visited)
            {
                match_col[v] = Some(u);
                return true;
            }
        }
    }
    false
}

/// Hungarian algorithm for minimum-cost perfect matching on a square cost matrix.
///
/// Returns assignment[i] = column matched to row i.
/// O(n^3) time, sufficient for our diagram sizes (typically < 500 points).
fn hungarian_algorithm(cost: &[Vec<f64>]) -> Vec<usize> {
    let n = cost.len();
    if n == 0 {
        return Vec::new();
    }

    // Use 1-indexed to simplify boundary conditions
    let mut u = vec![0.0_f64; n + 1]; // row potentials
    let mut v = vec![0.0_f64; n + 1]; // column potentials
    let mut assignment = vec![0_usize; n + 1]; // assignment[j] = row assigned to col j

    for i in 1..=n {
        let mut links = vec![0_usize; n + 1]; // links[j] = previous col in augmenting path
        let mut mins = vec![f64::INFINITY; n + 1]; // mins[j] = minimum reduced cost to col j
        let mut visited = vec![false; n + 1];

        // Start augmenting path from virtual column 0
        assignment[0] = i;
        let mut j0 = 0_usize;

        loop {
            visited[j0] = true;
            let i0 = assignment[j0];
            let mut delta = f64::INFINITY;
            let mut j1 = 0_usize;

            for j in 1..=n {
                if !visited[j] {
                    let reduced = cost[i0 - 1][j - 1] - u[i0] - v[j];
                    if reduced < mins[j] {
                        mins[j] = reduced;
                        links[j] = j0;
                    }
                    if mins[j] < delta {
                        delta = mins[j];
                        j1 = j;
                    }
                }
            }

            // Update potentials
            for j in 0..=n {
                if visited[j] {
                    u[assignment[j]] += delta;
                    v[j] -= delta;
                } else {
                    mins[j] -= delta;
                }
            }

            j0 = j1;

            if assignment[j0] == 0 {
                break;
            }
        }

        // Trace back augmenting path
        loop {
            let prev = links[j0];
            assignment[j0] = assignment[prev];
            j0 = prev;
            if j0 == 0 {
                break;
            }
        }
    }

    // Convert to 0-indexed: result[row] = col
    let mut result = vec![0_usize; n];
    for j in 1..=n {
        if assignment[j] > 0 {
            result[assignment[j] - 1] = j - 1;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simplex_boundary() {
        // Edge (0,1) has boundary {(0), (1)}
        let edge = Simplex::new(vec![0, 1], 1.0);
        let boundary = edge.boundary();
        assert_eq!(boundary.len(), 2);
        // Check vertices are present (order may vary after sorting)
        let verts: Vec<Vec<usize>> = boundary.iter().map(|s| s.vertices.clone()).collect();
        assert!(verts.contains(&vec![0]));
        assert!(verts.contains(&vec![1]));

        // Triangle (0,1,2) has boundary {(0,1), (0,2), (1,2)}
        let triangle = Simplex::new(vec![0, 1, 2], 2.0);
        let boundary = triangle.boundary();
        assert_eq!(boundary.len(), 3);
        assert!(boundary.iter().any(|s| s.vertices == vec![0, 1]));
        assert!(boundary.iter().any(|s| s.vertices == vec![0, 2]));
        assert!(boundary.iter().any(|s| s.vertices == vec![1, 2]));
    }

    #[test]
    fn test_distance_matrix_2d_triangle() {
        // Equilateral triangle in 2D: (0,0), (1,0), (0.5, sqrt(3)/2)
        let points = vec![
            0.0, 0.0, 0.0,  // vertex 0
            1.0, 0.0, 0.0,  // vertex 1
            0.5, 0.866, 0.0, // vertex 2 (approximately equilateral)
        ];
        let dist = DistanceMatrix::from_points_3d(&points);

        assert!((dist.get(0, 1) - 1.0).abs() < 1e-6);
        assert!((dist.get(0, 2) - 1.0).abs() < 0.01);
        assert!((dist.get(1, 2) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_vietoris_rips_triangle() {
        // Equilateral triangle
        let points = vec![
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            0.5, 0.866, 0.0,
        ];
        let dist = DistanceMatrix::from_points_3d(&points);
        let complex = VietorisRipsComplex::build(&dist, 1.5, 2);

        // Should have 3 vertices, 3 edges, 1 triangle
        assert_eq!(complex.simplices[0].len(), 3); // vertices
        assert_eq!(complex.simplices[1].len(), 3); // edges
        assert_eq!(complex.simplices[2].len(), 1); // triangle
    }

    #[test]
    fn test_persistent_homology_circle() {
        // Circle: 8 points on unit circle
        use std::f64::consts::PI;
        let n = 8;
        let mut points = Vec::with_capacity(n * 3);
        for i in 0..n {
            let theta = 2.0 * PI * (i as f64) / (n as f64);
            points.push(theta.cos());
            points.push(theta.sin());
            points.push(0.0);
        }

        let dist = DistanceMatrix::from_points_3d(&points);
        let complex = VietorisRipsComplex::build(&dist, 2.5, 2);
        let pairs = compute_persistent_homology(&complex);

        // Compute Betti numbers at late filtration time (after all edges form)
        let betti = compute_betti_numbers_at_time(&pairs, 1.5);

        // Circle should have b_0 = 1 (connected), b_1 = 1 (one loop)
        assert_eq!(betti[0], 1, "Circle should have 1 connected component at t=1.5");
        assert_eq!(betti[1], 1, "Circle should have 1 loop at t=1.5");
    }

    #[test]
    fn test_persistent_homology_two_components() {
        // Two separate points (disconnected)
        let points = vec![
            0.0, 0.0, 0.0,  // point 0
            10.0, 0.0, 0.0, // point 1 (far away)
        ];

        let dist = DistanceMatrix::from_points_3d(&points);
        let complex = VietorisRipsComplex::build(&dist, 5.0, 1);
        let pairs = compute_persistent_homology(&complex);

        // Should have 2 connected components (one dies when they connect)
        let betti = compute_betti_numbers(&pairs, 0.1);
        assert_eq!(betti[0], 2, "Two disconnected points should give b_0 = 2");
    }

    #[test]
    fn test_persistent_homology_tetrahedron() {
        // Regular tetrahedron (0 loops, but fills a 3D region)
        let s = 1.0 / 2.0_f64.sqrt();
        let points = vec![
            1.0, 0.0, -s,
            -1.0, 0.0, -s,
            0.0, 1.0, s,
            0.0, -1.0, s,
        ];

        let dist = DistanceMatrix::from_points_3d(&points);
        let complex = VietorisRipsComplex::build(&dist, 3.0, 2);
        let pairs = compute_persistent_homology(&complex);

        // Tetrahedron: b_0 = 1 (connected), b_1 = 0 (no loops)
        let betti = compute_betti_numbers_at_time(&pairs, 2.5);
        assert_eq!(betti[0], 1, "Tetrahedron should have 1 component at t=2.5");
        assert_eq!(betti[1], 0, "Tetrahedron should have 0 loops at t=2.5");
        // Note: b_2 requires 3-simplices in the complex
    }

    #[test]
    fn test_persistent_homology_sphere() {
        // Sphere: 12 points sampled from unit sphere surface
        let mut points = Vec::new();

        // Use icosahedron vertices (12 points on sphere)
        let phi = (1.0 + 5.0_f64.sqrt()) / 2.0; // golden ratio
        let vertices = [
            (0.0, 1.0, phi),
            (0.0, -1.0, phi),
            (0.0, 1.0, -phi),
            (0.0, -1.0, -phi),
            (1.0, phi, 0.0),
            (-1.0, phi, 0.0),
            (1.0, -phi, 0.0),
            (-1.0, -phi, 0.0),
            (phi, 0.0, 1.0),
            (-phi, 0.0, 1.0),
            (phi, 0.0, -1.0),
            (-phi, 0.0, -1.0),
        ];

        for (x, y, z) in &vertices {
            // Normalize to unit sphere
            let norm = (x * x + y * y + z * z).sqrt();
            points.push(x / norm);
            points.push(y / norm);
            points.push(z / norm);
        }

        let dist = DistanceMatrix::from_points_3d(&points);
        // Build complex up to dimension 2 (need triangles for H_2)
        let complex = VietorisRipsComplex::build(&dist, 3.0, 2);
        let pairs = compute_persistent_homology(&complex);

        // Sphere: b_0 = 1 (connected), b_1 = 0 (no 1D loops), b_2 = 1 (hollow shell)
        let betti = compute_betti_numbers_at_time(&pairs, 2.0);
        assert_eq!(betti[0], 1, "Sphere should have 1 component at t=2.0");
        assert_eq!(betti[1], 0, "Sphere should have 0 loops at t=2.0");
        // Note: Detecting b_2 requires careful threshold tuning and dense sampling
        // With 12 points (icosahedron), we might not detect the 2D void reliably
    }

    // -- Persistence diagram tests --

    #[test]
    fn test_persistence_diagram_from_pairs() {
        let pairs = vec![
            PersistencePair { dim: 0, birth: 0.0, death: 1.0 },
            PersistencePair { dim: 0, birth: 0.0, death: f64::INFINITY },
            PersistencePair { dim: 1, birth: 0.5, death: 2.0 },
        ];
        let dgm0 = PersistenceDiagram::from_pairs(&pairs, 0);
        assert_eq!(dgm0.len(), 1); // only finite pair
        assert_eq!(dgm0.dim, 0);
        assert!((dgm0.points[0].0 - 0.0).abs() < 1e-12);
        assert!((dgm0.points[0].1 - 1.0).abs() < 1e-12);

        let dgm1 = PersistenceDiagram::from_pairs(&pairs, 1);
        assert_eq!(dgm1.len(), 1);
        assert!((dgm1.points[0].0 - 0.5).abs() < 1e-12);
        assert!((dgm1.points[0].1 - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_persistence_diagram_all_dimensions() {
        let pairs = vec![
            PersistencePair { dim: 0, birth: 0.0, death: 1.0 },
            PersistencePair { dim: 1, birth: 0.3, death: 1.5 },
            PersistencePair { dim: 2, birth: 0.7, death: 2.0 },
        ];
        let diagrams = PersistenceDiagram::from_pairs_all(&pairs);
        assert_eq!(diagrams.len(), 3);
        assert_eq!(diagrams[0].len(), 1);
        assert_eq!(diagrams[1].len(), 1);
        assert_eq!(diagrams[2].len(), 1);
    }

    #[test]
    fn test_total_and_max_persistence() {
        let dgm = PersistenceDiagram {
            dim: 0,
            points: vec![(0.0, 1.0), (0.5, 2.0), (1.0, 1.5)],
        };
        assert!((dgm.total_persistence() - 3.0).abs() < 1e-12); // 1.0 + 1.5 + 0.5
        assert!((dgm.max_persistence() - 1.5).abs() < 1e-12);
    }

    #[test]
    fn test_persistence_entropy_uniform() {
        // Two features with equal persistence -> maximum entropy = ln(2)
        let dgm = PersistenceDiagram {
            dim: 0,
            points: vec![(0.0, 1.0), (0.0, 1.0)],
        };
        let h = dgm.persistence_entropy();
        assert!((h - 2.0_f64.ln()).abs() < 1e-12, "Uniform entropy should be ln(2)");
    }

    #[test]
    fn test_persistence_entropy_single() {
        // Single feature -> entropy = 0 (probability 1, ln(1) = 0)
        let dgm = PersistenceDiagram {
            dim: 0,
            points: vec![(0.0, 5.0)],
        };
        assert!((dgm.persistence_entropy() - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_persistence_entropy_empty() {
        let dgm = PersistenceDiagram { dim: 0, points: vec![] };
        assert!((dgm.persistence_entropy() - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_wasserstein_identical_diagrams() {
        let dgm = PersistenceDiagram {
            dim: 0,
            points: vec![(0.0, 1.0), (0.5, 2.0)],
        };
        let d = dgm.wasserstein_distance(&dgm, 2.0);
        assert!(d < 1e-10, "Distance between identical diagrams should be ~0, got {d}");
    }

    #[test]
    fn test_wasserstein_empty_vs_nonempty() {
        let empty = PersistenceDiagram { dim: 0, points: vec![] };
        let dgm = PersistenceDiagram {
            dim: 0,
            points: vec![(0.0, 2.0)],
        };
        // Cost = projecting (0,2) to diagonal = persistence/2 = 1.0
        let d = dgm.wasserstein_distance(&empty, 2.0);
        assert!((d - 1.0).abs() < 1e-6, "W2 to empty should be 1.0, got {d}");
    }

    #[test]
    fn test_wasserstein_symmetry() {
        let dgm1 = PersistenceDiagram {
            dim: 0,
            points: vec![(0.0, 1.0), (0.5, 3.0)],
        };
        let dgm2 = PersistenceDiagram {
            dim: 0,
            points: vec![(0.1, 1.2), (0.4, 2.8)],
        };
        let d12 = dgm1.wasserstein_distance(&dgm2, 2.0);
        let d21 = dgm2.wasserstein_distance(&dgm1, 2.0);
        assert!(
            (d12 - d21).abs() < 1e-10,
            "Wasserstein should be symmetric: {d12} vs {d21}"
        );
    }

    #[test]
    fn test_wasserstein_triangle_inequality() {
        let a = PersistenceDiagram {
            dim: 0,
            points: vec![(0.0, 1.0)],
        };
        let b = PersistenceDiagram {
            dim: 0,
            points: vec![(0.0, 2.0)],
        };
        let c = PersistenceDiagram {
            dim: 0,
            points: vec![(0.0, 3.0)],
        };
        let dab = a.wasserstein_distance(&b, 1.0);
        let dbc = b.wasserstein_distance(&c, 1.0);
        let dac = a.wasserstein_distance(&c, 1.0);
        assert!(
            dac <= dab + dbc + 1e-10,
            "Triangle inequality violated: d(a,c)={dac} > d(a,b)+d(b,c)={}", dab + dbc
        );
    }

    #[test]
    fn test_bottleneck_identical_diagrams() {
        let dgm = PersistenceDiagram {
            dim: 0,
            points: vec![(0.0, 1.0), (0.5, 2.0)],
        };
        let d = dgm.bottleneck_distance(&dgm);
        assert!(d < 1e-10, "Bottleneck between identical diagrams should be ~0, got {d}");
    }

    #[test]
    fn test_bottleneck_single_point_shift() {
        // One point shifted by delta in death coordinate
        let dgm1 = PersistenceDiagram {
            dim: 0,
            points: vec![(0.0, 1.0)],
        };
        let dgm2 = PersistenceDiagram {
            dim: 0,
            points: vec![(0.0, 1.5)],
        };
        let d = dgm1.bottleneck_distance(&dgm2);
        assert!(
            (d - 0.5).abs() < 1e-10,
            "Bottleneck should be 0.5 (death shift), got {d}"
        );
    }

    #[test]
    fn test_bottleneck_empty_vs_nonempty() {
        let empty = PersistenceDiagram { dim: 0, points: vec![] };
        let dgm = PersistenceDiagram {
            dim: 0,
            points: vec![(0.0, 4.0)],
        };
        // Bottleneck = persistence/2 = 2.0 (projecting to diagonal)
        let d = dgm.bottleneck_distance(&empty);
        assert!(
            (d - 2.0).abs() < 1e-10,
            "Bottleneck to empty should be 2.0, got {d}"
        );
    }

    #[test]
    fn test_hungarian_algorithm_basic() {
        // 2x2 cost matrix with obvious assignment
        let cost = vec![
            vec![1.0, 100.0],
            vec![100.0, 1.0],
        ];
        let result = hungarian_algorithm(&cost);
        assert_eq!(result[0], 0); // row 0 -> col 0
        assert_eq!(result[1], 1); // row 1 -> col 1
    }

    #[test]
    fn test_hungarian_algorithm_3x3() {
        // Classic assignment problem
        let cost = vec![
            vec![10.0, 5.0, 13.0],
            vec![3.0, 7.0, 15.0],
            vec![20.0, 11.0, 9.0],
        ];
        let result = hungarian_algorithm(&cost);
        // Optimal: row 0->col 1 (5), row 1->col 0 (3), row 2->col 2 (9) = 17
        let total: f64 = result.iter().enumerate().map(|(i, &j)| cost[i][j]).sum();
        assert!((total - 17.0).abs() < 1e-10, "Optimal cost should be 17, got {total}");
    }

    #[test]
    fn test_wasserstein_on_circle_vs_line() {
        // Circle (8 points) vs line (8 points) should have different H1 topology
        use std::f64::consts::PI;
        let n = 8;

        // Circle
        let mut circle_pts = Vec::with_capacity(n * 3);
        for i in 0..n {
            let theta = 2.0 * PI * (i as f64) / (n as f64);
            circle_pts.push(theta.cos());
            circle_pts.push(theta.sin());
            circle_pts.push(0.0);
        }
        let dist_circle = DistanceMatrix::from_points_3d(&circle_pts);
        let complex_circle = VietorisRipsComplex::build(&dist_circle, 2.5, 2);
        let pairs_circle = compute_persistent_homology(&complex_circle);
        let dgm_circle_h1 = PersistenceDiagram::from_pairs(&pairs_circle, 1);

        // Line segment
        let mut line_pts = Vec::with_capacity(n * 3);
        for i in 0..n {
            line_pts.push(i as f64 / (n as f64 - 1.0));
            line_pts.push(0.0);
            line_pts.push(0.0);
        }
        let dist_line = DistanceMatrix::from_points_3d(&line_pts);
        let complex_line = VietorisRipsComplex::build(&dist_line, 2.5, 2);
        let pairs_line = compute_persistent_homology(&complex_line);
        let dgm_line_h1 = PersistenceDiagram::from_pairs(&pairs_line, 1);

        let w2 = dgm_circle_h1.wasserstein_distance(&dgm_line_h1, 2.0);
        // Circle has a persistent H1 loop; line does not
        // So the Wasserstein distance should be positive
        assert!(
            w2 > 0.1,
            "Circle vs line H1 Wasserstein should be positive, got {w2}"
        );
    }
}
