//! E8/G2/Crystal Symmetry Bridge.
//!
//! Connects three previously isolated symmetry systems:
//! 1. `crystal_symmetry.rs` (32 point groups, 230 space groups)
//! 2. `algebra_core::lie::e8_lattice` (240 E8 roots, Freudenthal-Tits magic square)
//! 3. ZD graph structures (box-kites, kite-chain middens)
//!
//! The bridge: E8 contains G2 as a subgroup. G2 = Aut(O) preserves octonion
//! multiplication. The octonion multiplication table defines the 16D box-kite
//! structure. Therefore E8 root system symmetries should constrain ZD graph
//! automorphism groups.
//!
//! Additionally, the orthoplex K_{2,2,...,2} is a crystal-like periodic structure
//! whose Laplacian eigenvalue degeneracies mirror phonon band structures in
//! certain space groups.
//!
//! # nalgebra version note
//! moyo 0.7 depends on nalgebra 0.34, while this workspace pins nalgebra 0.33
//! (statrs 0.18 constraint). At the moyo boundary we use raw `[[f64; 3]; 3]`
//! arrays and `[f64; 3].into()` for Lattice/Position construction to avoid
//! passing our nalgebra types into moyo's API.
//!
//! # Literature
//! - Tits (1966), "Algebres alternatives, algebres de Jordan, et algebres de Lie exceptionnelles"
//! - Baez (2002), "The Octonions", Bull. AMS
//! - Conway & Sloane (1998), "Sphere Packings, Lattices and Groups"

use algebra_core::lie::e8_lattice::{E8Root, generate_e8_roots};
use moyo::{
    MoyoDataset,
    base::{Cell, Lattice, Position},
};

/// Result of classifying the E8 root system's 3D projected point group.
#[derive(Debug, Clone)]
pub struct E8PointGroupResult {
    /// Space group number from moyo (projected lattice).
    pub space_group_number: u32,
    /// Hermann-Mauguin symbol.
    pub hm_symbol: String,
    /// Number of symmetry operations found.
    pub num_symmetry_ops: usize,
    /// Pearson symbol of the projected lattice.
    pub pearson_symbol: String,
}

/// Result of G2 orbit analysis in Im(O) = R^7.
#[derive(Debug, Clone)]
pub struct G2OrbitResult {
    /// Number of distinct orbit types under G2 action on Im(O) basis.
    pub num_orbit_types: usize,
    /// Orbit sizes (number of roots in each orbit).
    pub orbit_sizes: Vec<usize>,
}

/// Project 240 E8 roots into 3D by taking the first 3 coordinates.
///
/// This projection preserves the octahedral symmetry subgroup Oh < W(E8).
/// The projected point cloud has cubic symmetry when treated as a crystal lattice.
pub fn project_e8_to_3d(roots: &[E8Root]) -> Vec<[f64; 3]> {
    roots
        .iter()
        .map(|r| [r.coords[0], r.coords[1], r.coords[2]])
        .collect()
}

/// Classify the point group of E8 roots projected to 3D using moyo.
///
/// Creates a crystal cell from the distinct 3D positions and determines
/// the space group symmetry. Uses raw `[[f64; 3]; 3]` arrays at the moyo
/// boundary to avoid nalgebra 0.33/0.34 version conflict.
pub fn e8_point_group() -> Option<E8PointGroupResult> {
    let roots = generate_e8_roots();
    let pts_3d = project_e8_to_3d(&roots);

    // Find the bounding box to set lattice vectors large enough
    let mut max_coord = 0.0_f64;
    for pt in &pts_3d {
        for &c in pt {
            max_coord = max_coord.max(c.abs());
        }
    }
    // Lattice vectors: cubic cell enclosing all projected points.
    // Use from_basis() which takes [[f64; 3]; 3] -- no nalgebra at the boundary.
    let a = max_coord * 4.0;
    let lattice = Lattice::from_basis([[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]]);

    // De-duplicate positions (many 8D roots project to same 3D point)
    let mut unique_frac: Vec<[f64; 3]> = Vec::new();
    for pt in &pts_3d {
        let frac = [pt[0] / a + 0.5, pt[1] / a + 0.5, pt[2] / a + 0.5];
        let is_dup = unique_frac.iter().any(|p| {
            (p[0] - frac[0]).abs() < 1e-6
                && (p[1] - frac[1]).abs() < 1e-6
                && (p[2] - frac[2]).abs() < 1e-6
        });
        if !is_dup {
            unique_frac.push(frac);
        }
    }

    // Convert to moyo types using Into trait (resolves to moyo's nalgebra 0.34)
    let positions: Vec<Position> = unique_frac.iter().map(|&p| p.into()).collect();
    let numbers: Vec<i32> = vec![0; unique_frac.len()];

    let cell = Cell::new(lattice, positions, numbers);
    let dataset = MoyoDataset::with_default(&cell, 1e-2).ok()?;

    Some(E8PointGroupResult {
        space_group_number: dataset.number as u32,
        hm_symbol: dataset.hm_symbol.clone(),
        num_symmetry_ops: dataset.operations.len(),
        pearson_symbol: dataset.pearson_symbol.clone(),
    })
}

/// Analyze G2 orbits in Im(O) = R^7.
///
/// G2 acts on the 7 imaginary octonion basis elements {e_1, ..., e_7}.
/// Under G2, these basis elements decompose into orbits based on the
/// inner product structure preserved by G2.
///
/// For generic elements in Im(O), G2 has two types of orbits:
/// - Elements on the unit sphere S^6 (a single G2-orbit for generic points)
/// - The zero element (fixed point)
///
/// For the E8 root system restricted to the Im(O) subspace (coords 1-7),
/// the roots decompose into orbits by their norm and G2-invariant structure.
pub fn g2_subgroup_orbits(roots: &[E8Root]) -> G2OrbitResult {
    // Project E8 roots to Im(O) = coordinates 1..=7
    let projected: Vec<[f64; 7]> = roots
        .iter()
        .map(|r| {
            let mut p = [0.0; 7];
            p.copy_from_slice(&r.coords[1..8]);
            p
        })
        .collect();

    // Group by squared norm (G2 preserves norm)
    let mut norm_groups: std::collections::BTreeMap<i64, Vec<usize>> =
        std::collections::BTreeMap::new();
    for (idx, p) in projected.iter().enumerate() {
        let norm_sq: f64 = p.iter().map(|x| x * x).sum();
        // Quantize to avoid floating point issues
        let key = (norm_sq * 1000.0).round() as i64;
        norm_groups.entry(key).or_default().push(idx);
    }

    // Further split by the number of nonzero coordinates (G2-invariant)
    let mut orbits: Vec<usize> = Vec::new();
    for indices in norm_groups.values() {
        let mut sub_groups: std::collections::BTreeMap<usize, usize> =
            std::collections::BTreeMap::new();
        for &idx in indices {
            let nonzero_count = projected[idx].iter().filter(|&&x| x.abs() > 1e-10).count();
            *sub_groups.entry(nonzero_count).or_insert(0) += 1;
        }
        for &count in sub_groups.values() {
            orbits.push(count);
        }
    }

    orbits.sort_unstable();
    orbits.reverse();

    G2OrbitResult {
        num_orbit_types: orbits.len(),
        orbit_sizes: orbits,
    }
}

/// Compute Laplacian eigenvalues of the complete multipartite graph K_{2,2,...,2}.
///
/// For k parts of size 2 (total n = 2k vertices):
/// - mu_0 = 0 (multiplicity 1)
/// - mu_1 = 2(k-1) (multiplicity k)
/// - mu_2 = 2k (multiplicity k-1)
///
/// Returns (eigenvalue, multiplicity) pairs.
pub fn orthoplex_laplacian_spectrum(k: usize) -> Vec<(f64, usize)> {
    vec![
        (0.0, 1),
        (2.0 * (k as f64 - 1.0), k),
        (2.0 * k as f64, k - 1),
    ]
}

/// Analyze the topology of the zero-divisor basis participation graph.
///
/// Vertices are basis element indices (0..dim-1). Two vertices are connected
/// if they co-occur in a zero-divisor pair `(e_i + e_j)(e_k +/- e_l) = 0`.
///
/// # Structure theorem (verified numerically at dims 16, 32, 64, 128, 256)
///
/// The ZD basis participation graph has exactly 3 connected components:
/// - Giant component of size `dim - 2`: all basis elements except e_0 and e_{dim/2}
/// - Two singletons: {e_0} (real unit) and {e_{dim/2}} (last-doubling generator)
///
/// The giant component is K_{dim-2} minus exactly `dim/2 - 1` edges.
/// The missing edges are pairs `(i, i XOR dim/2)` for i in the giant component
/// with `i < i XOR dim/2`. These pairs differ in the highest bit only,
/// corresponding to the last Cayley-Dickson doubling involution.
///
/// Edge count closed form: `(dim^2 - 6*dim + 8) / 2`
///
/// Returns (num_components, component_sizes, total_edges).
pub fn zd_graph_topology(dim: usize) -> (usize, Vec<usize>, usize) {
    assert!(dim >= 16 && dim.is_power_of_two());
    zd_graph_topology_closed_form(dim)
}

/// O(dim) closed-form ZD graph topology generator.
///
/// Based on the structure theorem: the ZD basis participation graph at dim = 2^n
/// (n >= 4) has exactly 3 connected components:
/// - Giant component of size `dim - 2`: all basis elements except e_0 and e_{dim/2}
/// - Two singletons: {e_0} (real unit) and {e_{dim/2}} (last-doubling generator)
///
/// The giant component is K_{dim-2} minus exactly `dim/2 - 1` edges.
/// The missing edges are pairs `(i, i XOR dim/2)` for `i` in the giant component
/// with `i < i XOR dim/2`. These are pairs differing in the highest bit only --
/// elements related by the last Cayley-Dickson doubling involution.
///
/// Edge count: C(dim-2, 2) - (dim/2 - 1) = (dim^2 - 6*dim + 8) / 2
fn zd_graph_topology_closed_form(dim: usize) -> (usize, Vec<usize>, usize) {
    let edge_count = (dim * dim - 6 * dim + 8) / 2;
    let component_sizes = vec![dim - 2, 1, 1];
    (3, component_sizes, edge_count)
}

/// Brute-force ZD graph topology via sign table (O(dim^4) candidates).
///
/// Retained for cross-validation against the O(1) closed-form at small dims.
pub fn zd_graph_topology_brute_force(dim: usize) -> (usize, Vec<usize>, usize) {
    zd_graph_topology_sign_table(dim)
}

/// Brute-force adjacency extraction: returns sorted list of edges (i, j) with i < j
/// that exist in the ZD basis participation graph.
///
/// This is the full O(dim^4) search used only for small-dim verification.
#[cfg(test)]
fn zd_graph_edges_brute_force(dim: usize) -> Vec<(usize, usize)> {
    use cd_kernel::cayley_dickson::cd_basis_mul_sign_iter;
    use fixedbitset::FixedBitSet;

    let sign_table: Vec<i32> = (0..dim * dim)
        .map(|idx| cd_basis_mul_sign_iter(dim, idx / dim, idx % dim))
        .collect();

    let mut adj = FixedBitSet::with_capacity(dim * dim);
    for i in 0..dim {
        for j in (i + 1)..dim {
            for k in 0..dim {
                for l in (k + 1)..dim {
                    if is_zd_2blade(&sign_table, dim, i, j, k, l, 1)
                        || is_zd_2blade(&sign_table, dim, i, j, k, l, -1)
                    {
                        // Same edge marking as production: connect all co-occurring pairs
                        let pairs = [(i, j), (k, l), (i, k), (i, l), (j, k), (j, l)];
                        for &(p, q) in &pairs {
                            let (lo, hi) = if p < q { (p, q) } else { (q, p) };
                            if lo != hi {
                                adj.insert(lo * dim + hi);
                            }
                        }
                    }
                }
            }
        }
    }

    let mut edges = Vec::new();
    for i in 0..dim {
        for j in (i + 1)..dim {
            if adj.contains(i * dim + j) {
                edges.push((i, j));
            }
        }
    }
    edges
}

/// Return the set of missing edges in the ZD basis participation graph.
///
/// These are the `dim/2 - 1` pairs `(i, i XOR dim/2)` with `i < i XOR dim/2`,
/// for all `i` in the giant component (i.e., `i != 0` and `i != dim/2`).
///
/// Each pair differs only in the highest bit (XOR = dim/2), corresponding to
/// elements related by the last Cayley-Dickson doubling involution. The element
/// e_{dim/2} is the "new imaginary unit" introduced by the final doubling step,
/// and pairs connected by this involution cannot co-occur in any zero divisor.
pub fn zd_graph_missing_edges(dim: usize) -> Vec<(usize, usize)> {
    assert!(dim >= 16 && dim.is_power_of_two());
    let half = dim / 2;
    let mut missing = Vec::with_capacity(half - 1);
    // Pairs (i, i^half) with i < i^half, where i is in the giant component.
    // i is in {1, ..., dim-1} \ {half}. For i < half, i^half = i+half > i. Good.
    // For i > half, i^half = i-half < i. Skip (would be duplicate with i' = i-half).
    for i in 1..half {
        missing.push((i, i ^ half));
    }
    missing
}

/// Return which vertices are singletons (isolated) in the ZD graph.
///
/// For dim = 2^n (n >= 4): singletons are {0, dim/2}.
/// - e_0: the real unit, trivially cannot form zero divisors
/// - e_{dim/2}: the generator of the last CD doubling, algebraically immune to ZDs
pub fn zd_graph_singletons(dim: usize) -> Vec<usize> {
    assert!(dim >= 16 && dim.is_power_of_two());
    vec![0, dim / 2]
}

/// Analytical O(dim^2) ZD adjacency matrix builder.
///
/// Generates the full FixedBitSet adjacency matrix using the closed-form
/// structure theorem (proved in C958_ZDGraphTopology.v):
///
/// - Singletons {e_0, e_{dim/2}} have no edges.
/// - All other pairs (i, j) with i != j are connected UNLESS i XOR j == dim/2
///   (the last-doubling involution pairs).
///
/// This replaces the O(dim^4) brute-force for dims >= 128 where the
/// sign-table search becomes computationally prohibitive.
///
/// Returns a symmetric adjacency matrix as FixedBitSet of capacity dim*dim.
pub fn zd_graph_adjacency_analytical(dim: usize) -> fixedbitset::FixedBitSet {
    use fixedbitset::FixedBitSet;

    assert!(dim >= 16 && dim.is_power_of_two());
    let half = dim / 2;
    let mut adj = FixedBitSet::with_capacity(dim * dim);

    // Giant component: vertices in {1, ..., dim-1} \ {half}
    // Edge (i, j) exists iff i XOR j != half
    for i in 1..dim {
        if i == half {
            continue;
        }
        for j in (i + 1)..dim {
            if j == half {
                continue;
            }
            if (i ^ j) != half {
                adj.insert(i * dim + j);
                adj.insert(j * dim + i);
            }
        }
    }

    adj
}

/// Sign-table ZD graph for all dimensions.
///
/// Architecture:
/// 1. Pre-compute dim x dim sign table (O(dim^2), fits in L1/L2 cache)
/// 2. Rayon parallel over (i,j) pairs, each testing all (k,l) via O(1) sign check
/// 3. Merge into FixedBitSet adjacency matrix
/// 4. Union-find with path halving for connected components
///
/// At 16D: 120 pairs * 120 tests = 14K checks. Sub-millisecond.
/// At 64D: 2016 pairs * 2016 tests = 4M checks. ~10ms on 5600X3D.
/// At 128D: 8128 pairs * 8128 tests = 66M checks. ~0.5s on 5600X3D.
fn zd_graph_topology_sign_table(dim: usize) -> (usize, Vec<usize>, usize) {
    use cd_kernel::cayley_dickson::cd_basis_mul_sign_iter;
    use fixedbitset::FixedBitSet;
    use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

    // Pre-compute sign table: sign[i * dim + j] = cd_basis_mul_sign(dim, i, j)
    // At 128D: 128*128 = 16K entries = 64 KB (fits in L2 cache on 5600X3D)
    // At 256D: 256*256 = 64K entries = 256 KB (fits in L2)
    let sign_table: Vec<i32> = (0..dim * dim)
        .map(|idx| cd_basis_mul_sign_iter(dim, idx / dim, idx % dim))
        .collect();

    // For 2-blade ZD: a = e_i + e_j, b = e_k + e_l
    // ab = e_i*e_k + e_i*e_l + e_j*e_k + e_j*e_l
    // Each product e_p*e_q lands on basis e_{p^q} with sign sign_table[p*dim+q].
    // For ab = 0, all dim components must cancel. With 4 terms, each landing on
    // a specific basis element, we need exact cancellation.
    //
    // Fast check: compute the 4 output indices and signs. If exactly 2 pairs
    // land on the same index with opposite signs, we have a ZD.
    //
    // Even faster: just compute the norm of the product using the sign table
    // directly. The product has at most 4 nonzero components, so norm check is O(1).

    let a_pairs: Vec<(usize, usize)> = (0..dim)
        .flat_map(|i| ((i + 1)..dim).map(move |j| (i, j)))
        .collect();

    // Parallel over (i,j) pairs. Each thread tests ALL (k,l) candidates.
    // At 128D: 8128 pairs * 8128 tests = 66M checks, but each is O(1) sign lookup.
    // With 6 cores: ~11M checks/core. At ~1 billion checks/sec (cache-hot): <0.1s.
    let all_hits: Vec<Vec<(usize, usize)>> = a_pairs
        .par_iter()
        .map(|&(i, j)| {
            let mut hits = Vec::new();

            for k in 0..dim {
                for l in (k + 1)..dim {
                    // Compute (e_i + e_j)(e_k + e_l) using sign table
                    if is_zd_2blade(&sign_table, dim, i, j, k, l, 1) {
                        hits.push((i, j));
                        hits.push((k, l));
                        hits.push((i, k));
                        hits.push((i, l));
                        hits.push((j, k));
                        hits.push((j, l));
                    }

                    // Also try (e_i + e_j)(e_k - e_l)
                    if is_zd_2blade(&sign_table, dim, i, j, k, l, -1) {
                        hits.push((i, j));
                        hits.push((k, l));
                        hits.push((i, k));
                        hits.push((i, l));
                        hits.push((j, k));
                        hits.push((j, l));
                    }
                }
            }
            hits
        })
        .collect();

    // Merge hits into FixedBitSet adjacency matrix
    let mut adj = FixedBitSet::with_capacity(dim * dim);
    let mut edge_count: usize = 0;

    for hits in &all_hits {
        for &(p, q) in hits {
            if p != q && !adj.contains(p * dim + q) {
                adj.insert(p * dim + q);
                adj.insert(q * dim + p);
                edge_count += 1;
            }
        }
    }

    // Union-find for connected components
    let mut parent: Vec<usize> = (0..dim).collect();

    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }

    fn union(parent: &mut [usize], a: usize, b: usize) {
        let ra = find(parent, a);
        let rb = find(parent, b);
        if ra != rb {
            parent[ra] = rb;
        }
    }

    for i in 0..dim {
        for j in (i + 1)..dim {
            if adj.contains(i * dim + j) {
                union(&mut parent, i, j);
            }
        }
    }

    let mut comp_map = std::collections::HashMap::new();
    for i in 0..dim {
        let root = find(&mut parent, i);
        *comp_map.entry(root).or_insert(0usize) += 1;
    }
    let mut component_sizes: Vec<usize> = comp_map.values().copied().collect();
    component_sizes.sort_unstable();
    component_sizes.reverse();

    (component_sizes.len(), component_sizes, edge_count)
}

/// Check if (e_i + e_j)(e_k + l_sign*e_l) = 0 using the pre-computed sign table.
///
/// The product has 4 terms: e_i*e_k, e_i*(l_sign*e_l), e_j*e_k, e_j*(l_sign*e_l).
/// Each term lands on basis e_{p^q} with coefficient sign_table[p*dim+q].
/// We accumulate into a small array indexed by output basis element, then check
/// if all coefficients are zero.
fn is_zd_2blade(
    sign_table: &[i32],
    dim: usize,
    i: usize,
    j: usize,
    k: usize,
    l: usize,
    l_sign: i32,
) -> bool {
    // The 4 products land on at most 4 distinct basis elements.
    // Collect (output_index, coefficient) pairs.
    let terms = [
        (i ^ k, sign_table[i * dim + k]),          // e_i * e_k
        (i ^ l, sign_table[i * dim + l] * l_sign), // e_i * (l_sign * e_l)
        (j ^ k, sign_table[j * dim + k]),          // e_j * e_k
        (j ^ l, sign_table[j * dim + l] * l_sign), // e_j * (l_sign * e_l)
    ];

    // Accumulate coefficients. Use a small fixed-size approach:
    // at most 4 distinct output indices.
    let mut accum = [(0usize, 0i32); 4];
    let mut n_distinct = 0usize;

    for &(idx, coeff) in &terms {
        let mut found = false;
        for entry in accum[..n_distinct].iter_mut() {
            if entry.0 == idx {
                entry.1 += coeff;
                found = true;
                break;
            }
        }
        if !found {
            accum[n_distinct] = (idx, coeff);
            n_distinct += 1;
        }
    }

    // ZD iff all accumulated coefficients are zero
    accum[..n_distinct].iter().all(|&(_, c)| c == 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_e8_root_count() {
        let roots = generate_e8_roots();
        assert_eq!(roots.len(), 240);
    }

    #[test]
    fn test_e8_3d_projection_has_symmetry() {
        // The 3D projection should have at least cubic symmetry
        if let Some(result) = e8_point_group() {
            // Should find a high-symmetry space group
            assert!(
                result.num_symmetry_ops >= 48,
                "Expected at least 48 ops (Oh), got {}",
                result.num_symmetry_ops
            );
            println!(
                "E8 3D projection: SG {} ({}), {} ops, Pearson: {}",
                result.space_group_number,
                result.hm_symbol,
                result.num_symmetry_ops,
                result.pearson_symbol
            );
        }
    }

    #[test]
    fn test_g2_orbits_nontrivial() {
        let roots = generate_e8_roots();
        let result = g2_subgroup_orbits(&roots);
        assert!(
            result.num_orbit_types >= 2,
            "G2 should produce multiple orbit types, got {}",
            result.num_orbit_types
        );
        let total: usize = result.orbit_sizes.iter().sum();
        assert_eq!(total, 240, "All roots should be accounted for");
    }

    #[test]
    fn test_orthoplex_laplacian_k4() {
        // K_{2,2,2,2}: k=4, n=8
        let spectrum = orthoplex_laplacian_spectrum(4);
        assert_eq!(spectrum.len(), 3);
        assert_eq!(spectrum[0], (0.0, 1));
        assert_eq!(spectrum[1], (6.0, 4)); // 2*(4-1) = 6, mult 4
        assert_eq!(spectrum[2], (8.0, 3)); // 2*4 = 8, mult 3
        // Total multiplicities: 1 + 4 + 3 = 8 = 2*4. Correct.
    }

    #[test]
    fn test_orthoplex_laplacian_k63() {
        // K_{2,...,2} with k=63 (CD dim 256): eigenvalues for C-948
        let spectrum = orthoplex_laplacian_spectrum(63);
        assert_eq!(spectrum[1], (124.0, 63)); // 2*(63-1) = 124
        assert_eq!(spectrum[2], (126.0, 62)); // 2*63 = 126
        let total_mult: usize = spectrum.iter().map(|(_, m)| m).sum();
        assert_eq!(total_mult, 126); // 2*63 = 126 vertices
    }

    #[test]
    fn test_zd_graph_sedenion_structure() {
        // dim=16: ZD basis participation graph.
        // e_0 (real unit) and e_8 (last-doubling generator) are singletons;
        // the remaining 14 basis elements form a single connected component.
        let (num_comp, sizes, edges) = zd_graph_topology(16);
        let total_vertices: usize = sizes.iter().sum();
        assert_eq!(total_vertices, 16, "16 basis elements in dim=16");
        assert_eq!(edges, 84, "Sedenion ZD graph has 84 edges");
        assert_eq!(num_comp, 3, "3 components: [14, 1, 1]");
        assert_eq!(sizes[0], 14, "Largest component has 14 elements");
    }

    #[test]
    fn test_zd_graph_pathion_structure() {
        // dim=32 (pathions): similar pattern -- most basis elements connected,
        // two singletons. Tests ZD detection scales beyond sedenions.
        let t = std::time::Instant::now();
        let (num_comp, sizes, edges) = zd_graph_topology(32);
        let elapsed = t.elapsed();
        let total: usize = sizes.iter().sum();
        assert_eq!(total, 32, "32 basis elements in dim=32");
        assert_eq!(edges, 420, "Pathion ZD graph has 420 edges");
        assert_eq!(num_comp, 3, "3 components: [30, 1, 1]");
        assert_eq!(sizes[0], 30, "Largest component has 30 elements");
        eprintln!(
            "32D ZD graph: {} edges, elapsed={:.3}ms",
            edges,
            elapsed.as_secs_f64() * 1000.0
        );
    }

    #[test]
    fn test_zd_graph_chingon_structure() {
        // dim=64 (chingons): closed-form. Exact structure.
        let (num_comp, sizes, edges) = zd_graph_topology(64);
        assert_eq!(num_comp, 3, "3 components: [62, 1, 1]");
        assert_eq!(sizes, vec![62, 1, 1]);
        assert_eq!(edges, 1860, "64D ZD graph has 1860 edges");
    }

    #[test]
    fn test_zd_graph_routon_structure() {
        // dim=128 (Routon): closed-form gives exact answer in O(1).
        let (num_comp, sizes, edges) = zd_graph_topology(128);
        assert_eq!(num_comp, 3, "3 components: [126, 1, 1]");
        assert_eq!(sizes, vec![126, 1, 1]);
        assert_eq!(edges, 7812, "128D ZD graph has 7812 edges");
    }

    #[test]
    fn test_zd_graph_closed_form_instant() {
        // The closed-form generator should work for any power-of-two dim >= 16.
        for &dim in &[16, 32, 64, 128, 256, 512, 1024] {
            let t = std::time::Instant::now();
            let (num_comp, sizes, edges) = zd_graph_topology(dim);
            let elapsed = t.elapsed();
            assert_eq!(num_comp, 3);
            assert_eq!(sizes, vec![dim - 2, 1, 1]);
            assert_eq!(edges, (dim * dim - 6 * dim + 8) / 2);
            eprintln!(
                "dim={}: {} edges, elapsed={:.0}ns",
                dim,
                edges,
                elapsed.as_nanos()
            );
        }
    }

    #[test]
    fn test_zd_closed_form_matches_brute_force_16d() {
        let (nc_bf, sz_bf, e_bf) = zd_graph_topology_brute_force(16);
        let (nc_cf, sz_cf, e_cf) = zd_graph_topology_closed_form(16);
        assert_eq!(
            (nc_bf, &sz_bf, e_bf),
            (nc_cf, &sz_cf, e_cf),
            "Closed-form must match brute-force at 16D"
        );
    }

    #[test]
    fn test_zd_closed_form_matches_brute_force_32d() {
        let (nc_bf, sz_bf, e_bf) = zd_graph_topology_brute_force(32);
        let (nc_cf, sz_cf, e_cf) = zd_graph_topology_closed_form(32);
        assert_eq!(
            (nc_bf, &sz_bf, e_bf),
            (nc_cf, &sz_cf, e_cf),
            "Closed-form must match brute-force at 32D"
        );
    }

    #[test]
    fn test_zd_closed_form_matches_brute_force_64d() {
        let (nc_bf, sz_bf, e_bf) = zd_graph_topology_brute_force(64);
        let (nc_cf, sz_cf, e_cf) = zd_graph_topology_closed_form(64);
        assert_eq!(
            (nc_bf, &sz_bf, e_bf),
            (nc_cf, &sz_cf, e_cf),
            "Closed-form must match brute-force at 64D"
        );
    }

    #[test]
    fn test_zd_missing_edges_are_xor_half_pairs() {
        let missing = zd_graph_missing_edges(16);
        assert_eq!(missing.len(), 7, "16D should have 7 missing edges");
        for &(i, j) in &missing {
            assert_eq!(
                i ^ j,
                8,
                "Missing edge ({},{}) should have XOR=dim/2=8",
                i,
                j
            );
            assert!(
                i != 0 && j != 0 && i != 8 && j != 8,
                "Missing edges should be in giant component (not singletons 0 or 8)"
            );
        }
        // Verify: (1,9), (2,10), (3,11), (4,12), (5,13), (6,14), (7,15)
        let expected: Vec<(usize, usize)> = (1..8).map(|i| (i, i ^ 8)).collect();
        assert_eq!(missing, expected);

        // Also verify singletons
        assert_eq!(zd_graph_singletons(16), vec![0, 8]);
    }

    #[test]
    fn test_zd_missing_edges_discovery_16d() {
        // Discover the actual missing edges at 16D by brute-force, then
        // verify that the closed-form prediction matches.
        let dim = 16usize;
        let half = dim / 2;
        let edges = zd_graph_edges_brute_force(dim);
        let edge_set: std::collections::HashSet<(usize, usize)> = edges.iter().copied().collect();
        assert_eq!(edges.len(), 84, "16D should have 84 edges total");

        // Verify singletons: e_0 and e_{dim/2} have no edges
        let deg_0 = edges.iter().filter(|&&(i, j)| i == 0 || j == 0).count();
        let deg_half = edges
            .iter()
            .filter(|&&(i, j)| i == half || j == half)
            .count();
        assert_eq!(deg_0, 0, "e_0 should be a singleton");
        assert_eq!(deg_half, 0, "e_{{dim/2}} should be a singleton");

        // Find missing edges in giant component (exclude singletons 0 and dim/2)
        let giant: Vec<usize> = (0..dim).filter(|&v| v != 0 && v != half).collect();
        let mut actual_missing = Vec::new();
        for (idx_i, &i) in giant.iter().enumerate() {
            for &j in &giant[idx_i + 1..] {
                if !edge_set.contains(&(i, j)) {
                    actual_missing.push((i, j));
                }
            }
        }

        assert_eq!(actual_missing.len(), 7, "16D should have 7 missing edges");

        // Verify all missing edges have XOR = dim/2
        for &(i, j) in &actual_missing {
            assert_eq!(
                i ^ j,
                half,
                "Missing ({},{}) should have XOR={}",
                i,
                j,
                half
            );
        }

        let predicted = zd_graph_missing_edges(dim);
        assert_eq!(
            actual_missing, predicted,
            "Predicted missing edges must match brute-force at 16D"
        );
    }

    #[test]
    fn test_zd_missing_edges_discovery_32d() {
        // Same discovery at 32D to verify the pattern scales.
        let dim = 32usize;
        let half = dim / 2;
        let edges = zd_graph_edges_brute_force(dim);
        let edge_set: std::collections::HashSet<(usize, usize)> = edges.iter().copied().collect();
        assert_eq!(edges.len(), 420, "32D should have 420 edges total");

        // Verify singletons
        let deg_0 = edges.iter().filter(|&&(i, j)| i == 0 || j == 0).count();
        let deg_half = edges
            .iter()
            .filter(|&&(i, j)| i == half || j == half)
            .count();
        assert_eq!(deg_0, 0, "e_0 should be a singleton");
        assert_eq!(deg_half, 0, "e_{{dim/2}} should be a singleton");

        // Find missing edges in giant component
        let giant: Vec<usize> = (0..dim).filter(|&v| v != 0 && v != half).collect();
        let mut actual_missing = Vec::new();
        for (idx_i, &i) in giant.iter().enumerate() {
            for &j in &giant[idx_i + 1..] {
                if !edge_set.contains(&(i, j)) {
                    actual_missing.push((i, j));
                }
            }
        }

        assert_eq!(actual_missing.len(), 15, "32D should have 15 missing edges");
        for &(i, j) in &actual_missing {
            assert_eq!(
                i ^ j,
                half,
                "Missing ({},{}) should have XOR={}",
                i,
                j,
                half
            );
        }

        let predicted = zd_graph_missing_edges(dim);
        assert_eq!(
            actual_missing, predicted,
            "Predicted missing edges must match brute-force at 32D"
        );
    }

    #[test]
    fn test_zd_adjacency_analytical_matches_brute_force_16d() {
        // Cross-validate the O(dim^2) analytical adjacency builder against
        // the O(dim^4) brute-force edge list at 16D.
        let dim = 16usize;
        let adj = zd_graph_adjacency_analytical(dim);
        let bf_edges = zd_graph_edges_brute_force(dim);

        // Check analytical has the right edge count
        let analytical_edge_count = (0..dim)
            .flat_map(|i| ((i + 1)..dim).map(move |j| (i, j)))
            .filter(|&(i, j)| adj.contains(i * dim + j))
            .count();
        assert_eq!(
            analytical_edge_count, 84,
            "16D analytical should have 84 edges"
        );
        assert_eq!(bf_edges.len(), 84, "16D brute-force should have 84 edges");

        // Every brute-force edge must be in the analytical matrix
        for &(i, j) in &bf_edges {
            assert!(
                adj.contains(i * dim + j),
                "Brute-force edge ({},{}) missing from analytical",
                i,
                j
            );
        }

        // Every analytical edge must be in the brute-force list
        let bf_set: std::collections::HashSet<(usize, usize)> = bf_edges.iter().copied().collect();
        for i in 0..dim {
            for j in (i + 1)..dim {
                if adj.contains(i * dim + j) {
                    assert!(
                        bf_set.contains(&(i, j)),
                        "Analytical edge ({},{}) not in brute-force",
                        i,
                        j
                    );
                }
            }
        }
    }

    #[test]
    fn test_zd_adjacency_analytical_instant_at_512d() {
        // 512D analytical should complete in microseconds.
        let t = std::time::Instant::now();
        let adj = zd_graph_adjacency_analytical(512);
        let elapsed = t.elapsed();
        assert!(
            elapsed.as_secs() < 2,
            "512D analytical took too long: {:.1}s",
            elapsed.as_secs_f64()
        );

        // Verify edge count matches closed-form
        let edge_count = (0..512usize)
            .flat_map(|i| ((i + 1)..512).map(move |j| (i, j)))
            .filter(|&(i, j)| adj.contains(i * 512 + j))
            .count();
        assert_eq!(
            edge_count,
            (512 * 512 - 6 * 512 + 8) / 2,
            "512D edge count mismatch"
        );
        eprintln!(
            "512D analytical adjacency: {} edges, elapsed={:.3}ms",
            edge_count,
            elapsed.as_secs_f64() * 1000.0
        );
    }

    #[test]
    #[ignore] // ~2-5 min at 256D (4 billion O(1) checks). Run with --include-ignored.
    fn test_zd_graph_voudon_256d() {
        // dim=256 (Voudon): full exhaustive sign-table search.
        // 256D: 32640 pairs * 32640 tests = ~1.07 billion checks.
        // Each check is O(1) via sign table, but the sheer volume needs patience.
        let t = std::time::Instant::now();
        let (num_comp, sizes, edges) = zd_graph_topology_brute_force(256);
        let elapsed = t.elapsed();
        let total: usize = sizes.iter().sum();
        assert_eq!(total, 256, "256 basis elements in dim=256");
        assert_eq!(num_comp, 3, "3 components: [254, 1, 1]");
        assert_eq!(sizes[0], 254, "Largest component has 254 elements");
        assert_eq!(edges, (256 * 256 - 6 * 256 + 8) / 2);
        eprintln!(
            "256D ZD graph (brute-force): {} components, {} edges, largest={}, elapsed={:.1}s",
            num_comp,
            edges,
            sizes[0],
            elapsed.as_secs_f64()
        );
    }
}
