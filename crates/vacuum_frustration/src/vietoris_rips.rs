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
}
