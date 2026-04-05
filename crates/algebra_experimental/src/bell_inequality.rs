use cd_kernel::cayley_dickson::{SignTable, cd_multiply, cd_norm_sq};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use std::collections::HashSet;

/// Result of a Bell-Inequality (CHSH) violation test.
#[derive(Debug, Clone)]
pub struct BellTestResult {
    pub dimension: usize,
    pub s_value: f64,
    pub violates_classical: bool,
    /// Individual correlator values: E(a,b), E(a,b'), E(a',b), E(a',b')
    pub correlators: [f64; 4],
    /// Number of ZD pairs used as entanglement channels
    pub n_zd_pairs: usize,
    /// Number of channels with valid probes (usable for CHSH)
    pub n_usable_channels: usize,
    /// Number of shared ZD paths found (connectivity)
    pub n_shared_paths: usize,
    /// Number of optimal probes (valid across all 4 CHSH angle combos)
    pub n_optimal_probes: usize,
    /// Statistical uncertainty on S (standard error)
    pub s_error: f64,
}

/// A zero-divisor pair lifted to high dimension.
#[derive(Debug, Clone)]
pub struct ZdChannel {
    // Dense representation retained for test validation (cd_multiply cross-check).
    #[allow(dead_code)]
    pub a: Vec<f64>,
    #[allow(dead_code)]
    pub b: Vec<f64>,
    /// Sparse representation of a: [(basis_idx, coefficient), ...]
    pub a_sparse: Vec<(usize, f64)>,
    /// Sparse representation of b: [(basis_idx, coefficient), ...]
    pub b_sparse: Vec<(usize, f64)>,
    /// Basis indices with non-zero support in a or b
    pub basis_indices: Vec<usize>,
    /// Optimal probes: basis elements where the associator [A', e_k, B'] is non-zero
    /// across ALL four (Alice plane, Bob plane) basis combinations.
    /// These are X_optimal = X_{a1,b1} AND X_{a1,b2} AND X_{a2,b1} AND X_{a2,b2}.
    pub optimal_probes: Vec<usize>,
    /// Union probes: basis elements where at least one [e_ai, e_k, e_bj] is non-zero.
    /// Superset of optimal_probes; used as fallback when optimal set is empty.
    pub union_probes: Vec<usize>,
}

/// Lift a sedenion (16D) 2-blade ZD pair to target dimension via Cayley-Dickson doubling.
///
/// Embedding: A_512 = (A_16, 0, 0, ..., 0). Because the right-hand elements are zero,
/// conjugate terms in the doubling formula drop out, and A*B = 0 is strictly preserved.
///
/// Uses the pre-computed sign table for sparse probe finding, avoiding the O(dim^2)
/// dense cd_associator that was the 512D bottleneck.
pub fn lift_zd_to_dim(
    a_16: &[f64; 16],
    b_16: &[f64; 16],
    target_dim: usize,
    alice_plane: (usize, usize),
    bob_plane: (usize, usize),
    sign_cache: &SignTableCache,
) -> ZdChannel {
    assert!(target_dim >= 16 && target_dim.is_power_of_two());

    let mut a = vec![0.0; target_dim];
    let mut b = vec![0.0; target_dim];

    a[..16].copy_from_slice(a_16);
    b[..16].copy_from_slice(b_16);

    let mut basis_set = HashSet::new();
    for (i, &v) in a.iter().enumerate() {
        if v.abs() > 1e-15 {
            basis_set.insert(i);
        }
    }
    for (i, &v) in b.iter().enumerate() {
        if v.abs() > 1e-15 {
            basis_set.insert(i);
        }
    }
    let basis_indices: Vec<usize> = basis_set.into_iter().collect();

    let a_sparse = to_sparse(&a);
    let b_sparse = to_sparse(&b);

    // Compute probe validity using the sparse sign-table path.
    // For Alice rotating in plane (a1, a2), her rotated element is:
    //   A' = cos(theta_a) * e_{a1} + sin(theta_a) * e_{a2}  (within her support)
    // For non-trivial correlations at ALL angles, X must be valid for ALL four combos.
    let (a1, a2) = alice_plane;
    let (b1, b2) = bob_plane;

    let probes_a1_b1 =
        find_valid_probes_sparse(&a_sparse, &b_sparse, a1, b1, target_dim, sign_cache);
    let probes_a1_b2 =
        find_valid_probes_sparse(&a_sparse, &b_sparse, a1, b2, target_dim, sign_cache);
    let probes_a2_b1 =
        find_valid_probes_sparse(&a_sparse, &b_sparse, a2, b1, target_dim, sign_cache);
    let probes_a2_b2 =
        find_valid_probes_sparse(&a_sparse, &b_sparse, a2, b2, target_dim, sign_cache);

    // X_optimal = intersection of all four sets
    let optimal_probes: Vec<usize> = probes_a1_b1
        .iter()
        .copied()
        .filter(|k| {
            probes_a1_b2.contains(k) && probes_a2_b1.contains(k) && probes_a2_b2.contains(k)
        })
        .collect();

    // X_union = union of all four sets (fallback)
    let mut union_set = HashSet::new();
    for &k in probes_a1_b1
        .iter()
        .chain(probes_a1_b2.iter())
        .chain(probes_a2_b1.iter())
        .chain(probes_a2_b2.iter())
    {
        union_set.insert(k);
    }
    let union_probes: Vec<usize> = union_set.into_iter().collect();

    ZdChannel {
        a,
        b,
        a_sparse,
        b_sparse,
        basis_indices,
        optimal_probes,
        union_probes,
    }
}

/// Find basis elements e_k where [A_rotated_in_axis_i, e_k, B_rotated_in_axis_j] != 0.
///
/// Uses the sparse sign-table path for O(support^2) per probe instead of O(dim^2).
/// This is critical at 512D+ where dense cd_associator is too slow.
pub fn find_valid_probes_sparse(
    a_sparse: &[(usize, f64)],
    b_sparse: &[(usize, f64)],
    alice_axis: usize,
    bob_axis: usize,
    dim: usize,
    sign_cache: &SignTableCache,
) -> Vec<usize> {
    let theta_test = 0.3;
    let next_a = next_axis(alice_axis, dim);
    let next_b = next_axis(bob_axis, dim);
    let a_rotated = rotate_sparse(a_sparse, alice_axis, next_a, theta_test);
    let b_rotated = rotate_sparse(b_sparse, bob_axis, next_b, theta_test);

    // Compute support bound from sparse elements
    let max_a = a_rotated.iter().map(|&(i, _)| i).max().unwrap_or(0) + 1;
    let max_b = b_rotated.iter().map(|&(i, _)| i).max().unwrap_or(0) + 1;
    let max_support = max_a.max(max_b);
    let scan_limit = dim.min(max_support.next_power_of_two() * 2);

    (0..scan_limit)
        .filter(|&k| {
            let x_sparse = [(k, 1.0)];
            let total = sign_cache.sparse_associator_sum(&a_rotated, &x_sparse, &b_rotated);
            total.abs() > 1e-20
        })
        .collect()
}

/// Get a companion axis for rotation: the other axis in the pair.
pub fn next_axis(axis: usize, dim: usize) -> usize {
    if axis + 1 < dim { axis + 1 } else { 1 }
}

/// Known sedenion 2-blade zero-divisor pairs from the box-kite structure.
///
/// In sedenions (dim=16), the box-kite construction (de Marrais 2001) produces
/// ZD pairs. Each is a 2-blade: (e_i + e_j) * (e_k +/- e_l) = 0.
///
/// Generated by brute-force enumeration at dim=16 (only ~16K checks).
pub fn known_sedenion_zd_pairs() -> Vec<([f64; 16], [f64; 16])> {
    let dim = 16;
    let mut pairs = Vec::new();

    for i in 1..dim {
        for j in (i + 1)..dim {
            let mut a = [0.0f64; 16];
            a[i] = 1.0;
            a[j] = 1.0;

            for k in 1..dim {
                for l in (k + 1)..dim {
                    let mut b = [0.0f64; 16];
                    b[k] = 1.0;
                    b[l] = 1.0;
                    let ab = cd_multiply(&a[..], &b[..]);
                    let norm = cd_norm_sq(&ab).sqrt();
                    if norm < 1e-12 {
                        pairs.push((a, b));
                    }

                    // Try e_k - e_l
                    b[l] = -1.0;
                    let ab2 = cd_multiply(&a[..], &b[..]);
                    let norm2 = cd_norm_sq(&ab2).sqrt();
                    if norm2 < 1e-12 {
                        pairs.push((a, b));
                    }
                }
            }
        }
    }

    pairs
}

/// Rotate element `x` in the (axis_a, axis_b) SO(2) subplane by angle `theta`.
///
/// The rotation is strictly confined to this 2D subplane, preventing the
/// rotated element from escaping the ZD graph structure:
///   x'[axis_a] = x[axis_a] * cos(theta) - x[axis_b] * sin(theta)
///   x'[axis_b] = x[axis_a] * sin(theta) + x[axis_b] * cos(theta)
#[allow(dead_code)]
fn rotate_in_plane(x: &[f64], axis_a: usize, axis_b: usize, theta: f64) -> Vec<f64> {
    let mut result = x.to_vec();
    let (c, s) = (theta.cos(), theta.sin());
    let xa = x[axis_a];
    let xb = x[axis_b];
    result[axis_a] = xa * c - xb * s;
    result[axis_b] = xa * s + xb * c;
    result
}

/// Pre-computed sign table for O(1) basis multiplication.
///
/// Precomputed sign table for sparse CD arithmetic.
///
/// Graduated from local SignTableCache to cd_kernel::SignTable (bitvec, 8x more
/// memory-efficient: 32 KB at 512D vs 1 MB for the former Vec<i32> layout).
/// `sparse_multiply` and `sparse_associator_sum` live on SignTable in cd_kernel.
pub type SignTableCache = SignTable;

/// Extract sparse representation from a dense vector.
/// Delegates to `SignTable::to_sparse`; kept for call-site compatibility.
pub fn to_sparse(v: &[f64]) -> Vec<(usize, f64)> {
    SignTable::to_sparse(v)
}

/// Rotate sparse element in the (axis_a, axis_b) SO(2) subplane by angle theta.
///
/// Only modifies the two affected axes, preserving sparsity.
pub fn rotate_sparse(
    sparse: &[(usize, f64)],
    axis_a: usize,
    axis_b: usize,
    theta: f64,
) -> Vec<(usize, f64)> {
    let (c, s) = (theta.cos(), theta.sin());

    // Find current values at the rotation axes
    let va = sparse
        .iter()
        .find(|&&(i, _)| i == axis_a)
        .map_or(0.0, |&(_, v)| v);
    let vb = sparse
        .iter()
        .find(|&&(i, _)| i == axis_b)
        .map_or(0.0, |&(_, v)| v);

    let new_a = va * c - vb * s;
    let new_b = va * s + vb * c;

    let mut result: Vec<(usize, f64)> = sparse
        .iter()
        .filter(|&&(i, _)| i != axis_a && i != axis_b)
        .copied()
        .collect();

    if new_a.abs() > 1e-15 {
        result.push((axis_a, new_a));
    }
    if new_b.abs() > 1e-15 {
        result.push((axis_b, new_b));
    }
    result
}

/// Compute the associator-based measurement outcome using sparse sign-table arithmetic.
///
/// Given a ZD channel (A, B), measurement angle theta for the measured party,
/// rotation plane, and probe element e_k:
///   outcome = sign( sum_j [A'(theta), e_k, B]_j )
///
/// The probe e_k must be valid (non-zero associator with both A and B sides).
///
/// This is the FAST PATH: uses pre-computed sign table for O(support^2) evaluation
/// instead of O(dim^2) recursive cd_multiply. For sedenion-lifted elements with
/// ~4 nonzero components, this is ~16 sign lookups vs ~262K recursive multiplies at 512D.
pub fn associator_measurement_fast(
    measured_sparse: &[(usize, f64)],
    partner_sparse: &[(usize, f64)],
    theta: f64,
    plane: (usize, usize),
    probe_idx: usize,
    sign_cache: &SignTableCache,
) -> f64 {
    let rotated = rotate_sparse(measured_sparse, plane.0, plane.1, theta);
    let x_sparse = [(probe_idx, 1.0)];

    let total = sign_cache.sparse_associator_sum(&rotated, &x_sparse, partner_sparse);

    if total >= 0.0 { 1.0 } else { -1.0 }
}

/// CHSH measurement configuration.
struct ChshConfig {
    alice_plane: (usize, usize),
    bob_plane: (usize, usize),
    n_trials: usize,
}

/// Compute the CHSH correlator E(theta_a, theta_b) using sparse sign-table arithmetic.
///
/// For each trial:
/// 1. Pick a random ZD channel (shared entanglement resource)
/// 2. Pick a random *optimal* probe (valid across all angle combos; falls back to union)
/// 3. Alice: sign([A'(theta_a), e_k, B])
/// 4. Bob: sign([B'(theta_b), e_k, A])
/// 5. Record product of outcomes
///
/// E = <outcome_a * outcome_b> averaged over n_trials.
///
/// Performance: O(support^2) per measurement via pre-computed sign table.
/// At 512D with ~4-component sparse elements: ~16 sign lookups vs ~262K
/// recursive multiplies per measurement.
fn compute_chsh_correlator(
    channels: &[ZdChannel],
    theta_a: f64,
    theta_b: f64,
    cfg: &ChshConfig,
    sign_cache: &SignTableCache,
    rng: &mut ChaCha8Rng,
) -> f64 {
    let usable: Vec<&ZdChannel> = channels
        .iter()
        .filter(|ch| !ch.optimal_probes.is_empty() || !ch.union_probes.is_empty())
        .collect();

    if usable.is_empty() {
        return 0.0;
    }

    let mut total = 0.0;

    for _ in 0..cfg.n_trials {
        let ch = usable[rng.random_range(0..usable.len())];

        // Prefer optimal probes; fall back to union if optimal is empty
        let probes = if !ch.optimal_probes.is_empty() {
            &ch.optimal_probes
        } else {
            &ch.union_probes
        };
        let probe = probes[rng.random_range(0..probes.len())];

        // Alice measures: sign([A'(theta_a), e_probe, B])
        let outcome_a = associator_measurement_fast(
            &ch.a_sparse,
            &ch.b_sparse,
            theta_a,
            cfg.alice_plane,
            probe,
            sign_cache,
        );

        // Bob measures: sign([B'(theta_b), e_probe, A])
        let outcome_b = associator_measurement_fast(
            &ch.b_sparse,
            &ch.a_sparse,
            theta_b,
            cfg.bob_plane,
            probe,
            sign_cache,
        );

        total += outcome_a * outcome_b;
    }

    total / cfg.n_trials as f64
}

/// Run CHSH inequality test using associator torque correlations in the CD algebra.
///
/// The CHSH inequality: for any local hidden variable theory, |S| <= 2.
/// Quantum mechanics (Tsirelson bound): |S| <= 2*sqrt(2) ~ 2.828.
///
/// We test whether Cayley-Dickson non-associativity produces S > 2.
///
/// The correlation mechanism: Alice and Bob share a ZD pair (A, B) where A*B ~ 0.
/// Each rotates their element in an SO(2) subplane and measures the sign of the
/// associator torque through a shared probe element X. The associator [A', X, B']
/// is a genuinely non-factorizable trilinear map -- it cannot be decomposed as
/// f(A') * g(B') for any functions f, g. This is the algebraic analogue of
/// quantum entanglement.
pub fn chsh_violation_test(dim: usize, n_samples: usize, seed: u64) -> BellTestResult {
    assert!(dim >= 16 && dim.is_power_of_two());

    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Measurement planes: orthogonal SO(2) subplanes within the sedenion support.
    // Alice rotates in (e_1, e_2), Bob in (e_3, e_4). These are within the first
    // 16 basis elements where the lifted ZD pairs have support.
    let alice_plane = (1, 2);
    let bob_plane = (3, 4);

    // Step 1: Pre-compute sign table for O(1) basis multiplication.
    // At 512D: 512*512 = 262K entries = 1 MB. Fits in L2 cache on 5600X3D.
    // Built FIRST so lift_zd_to_dim can use sparse probe finding.
    let sign_cache = SignTableCache::new(dim);

    // Step 2: Get sedenion ZD pairs and lift to target dimension
    let sed_pairs = known_sedenion_zd_pairs();
    let channels: Vec<ZdChannel> = sed_pairs
        .iter()
        .map(|(a, b)| lift_zd_to_dim(a, b, dim, alice_plane, bob_plane, &sign_cache))
        .collect();

    let n_zd_pairs = channels.len();
    let n_usable = channels
        .iter()
        .filter(|ch| !ch.optimal_probes.is_empty() || !ch.union_probes.is_empty())
        .count();
    let n_optimal_probes: usize = channels.iter().map(|ch| ch.optimal_probes.len()).sum();

    // Step 3: CHSH measurement settings (optimal for max quantum violation)
    let a = 0.0;
    let a_prime = std::f64::consts::FRAC_PI_2;
    let b = std::f64::consts::FRAC_PI_4;
    let b_prime = -std::f64::consts::FRAC_PI_4;

    // Step 4: Compute all four correlators via sparse sign-table arithmetic
    let cfg = ChshConfig {
        alice_plane,
        bob_plane,
        n_trials: n_samples,
    };

    let e_ab = compute_chsh_correlator(&channels, a, b, &cfg, &sign_cache, &mut rng);
    let e_ab_prime = compute_chsh_correlator(&channels, a, b_prime, &cfg, &sign_cache, &mut rng);
    let e_a_prime_b = compute_chsh_correlator(&channels, a_prime, b, &cfg, &sign_cache, &mut rng);
    let e_a_prime_b_prime =
        compute_chsh_correlator(&channels, a_prime, b_prime, &cfg, &sign_cache, &mut rng);

    // Step 5: S-value
    let s_value = e_ab - e_ab_prime + e_a_prime_b + e_a_prime_b_prime;

    // Step 6: Statistical error (SE for binary outcomes)
    let s_error = 4.0 / (n_samples as f64).sqrt();

    // Step 7: Shared ZD paths
    let n_shared_paths = count_shared_zd_paths(&channels);

    BellTestResult {
        dimension: dim,
        s_value,
        violates_classical: s_value.abs() > 2.0,
        correlators: [e_ab, e_ab_prime, e_a_prime_b, e_a_prime_b_prime],
        n_zd_pairs,
        n_usable_channels: n_usable,
        n_shared_paths,
        n_optimal_probes,
        s_error,
    }
}

/// Check if two sorted slices share any element (merge-join, zero allocation).
///
/// # Why sorted-Vec merge instead of HashSet
///
/// The old implementation allocated two HashSets per channel pair in a double
/// loop -- O(n^2) allocations with ~64 bytes overhead per entry. Sorted-Vec
/// merge-join uses zero allocation and O(k1 + k2) time via two-pointer scan.
/// Pattern from `algebra_analysis::sparse::SparseState::dot()`.
fn sorted_slices_intersect(a: &[usize], b: &[usize]) -> bool {
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => return true,
        }
    }
    false
}

/// Count shared ZD paths: pairs of channels with overlapping basis support.
///
/// Uses sorted-Vec merge-join intersection (zero per-pair allocation).
fn count_shared_zd_paths(channels: &[ZdChannel]) -> usize {
    // Pre-sort each channel's basis_indices once
    let sorted: Vec<Vec<usize>> = channels
        .iter()
        .map(|ch| {
            let mut v = ch.basis_indices.clone();
            v.sort_unstable();
            v.dedup();
            v
        })
        .collect();

    let mut count = 0;
    for (i, si) in sorted.iter().enumerate() {
        for sj in sorted.iter().skip(i + 1) {
            if sorted_slices_intersect(si, sj) {
                count += 1;
            }
        }
    }
    count
}

/// Find connected components of ZD channels linked by shared basis elements.
///
/// Each component represents a family of ZD pairs that can influence each other
/// through the associator torque -- the non-local correlation topology.
pub fn find_shared_zd_paths(dim: usize) -> Vec<Vec<usize>> {
    assert!(dim >= 16 && dim.is_power_of_two());

    let alice_plane = (1, 2);
    let bob_plane = (3, 4);

    let sign_cache = SignTableCache::new(dim);
    let sed_pairs = known_sedenion_zd_pairs();
    let channels: Vec<ZdChannel> = sed_pairs
        .iter()
        .map(|(a, b)| lift_zd_to_dim(a, b, dim, alice_plane, bob_plane, &sign_cache))
        .collect();

    let n = channels.len();
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];

    // Pre-sort basis_indices for merge-join intersection (zero per-pair allocation)
    let sorted: Vec<Vec<usize>> = channels
        .iter()
        .map(|ch| {
            let mut v = ch.basis_indices.clone();
            v.sort_unstable();
            v.dedup();
            v
        })
        .collect();

    for (i, si) in sorted.iter().enumerate() {
        for (j, sj) in sorted.iter().enumerate().skip(i + 1) {
            if sorted_slices_intersect(si, sj) {
                adj[i].push(j);
                adj[j].push(i);
            }
        }
    }

    // BFS connected components
    let mut visited = vec![false; n];
    let mut components = Vec::new();

    for start in 0..n {
        if visited[start] {
            continue;
        }
        let mut component = Vec::new();
        let mut stack = vec![start];
        while let Some(node) = stack.pop() {
            if !visited[node] {
                visited[node] = true;
                component.push(node);
                for &neighbor in &adj[node] {
                    if !visited[neighbor] {
                        stack.push(neighbor);
                    }
                }
            }
        }
        component.sort();
        components.push(component);
    }

    components
}

/// Dimensional scaling: run CHSH at dims 16, 32, ..., max_dim.
pub fn chsh_dimensional_scaling(
    max_dim: usize,
    n_samples: usize,
    seed: u64,
) -> Vec<(usize, BellTestResult)> {
    let mut results = Vec::new();
    let mut dim = 16;
    while dim <= max_dim {
        let result = chsh_violation_test(dim, n_samples, seed);
        results.push((dim, result));
        dim *= 2;
    }
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sedenion_zd_pairs_found() {
        let pairs = known_sedenion_zd_pairs();
        assert!(
            !pairs.is_empty(),
            "Sedenions must have 2-blade zero-divisor pairs"
        );
        println!("Found {} sedenion ZD pairs", pairs.len());
    }

    #[test]
    fn test_zd_pairs_are_genuine() {
        let pairs = known_sedenion_zd_pairs();
        for (i, (a, b)) in pairs.iter().enumerate().take(10) {
            let ab = cd_multiply(&a[..], &b[..]);
            let norm = cd_norm_sq(&ab).sqrt();
            assert!(
                norm < 1e-10,
                "ZD pair {} has ||a*b|| = {}, expected ~0",
                i,
                norm
            );
        }
    }

    #[test]
    fn test_lift_preserves_zd() {
        let pairs = known_sedenion_zd_pairs();
        assert!(!pairs.is_empty());

        let (a16, b16) = &pairs[0];
        let alice_plane = (1, 2);
        let bob_plane = (3, 4);
        for &target_dim in &[32, 64, 128, 256, 512] {
            let sc = SignTableCache::new(target_dim);
            let ch = lift_zd_to_dim(a16, b16, target_dim, alice_plane, bob_plane, &sc);
            let ab = cd_multiply(&ch.a, &ch.b);
            let norm = cd_norm_sq(&ab).sqrt();
            assert!(
                norm < 1e-10,
                "Lifted ZD at dim={} has ||a*b|| = {}, expected ~0",
                target_dim,
                norm
            );
        }
    }

    #[test]
    fn test_valid_probes_exist() {
        let pairs = known_sedenion_zd_pairs();
        assert!(!pairs.is_empty());

        let (a16, b16) = &pairs[0];
        let sc = SignTableCache::new(16);
        let ch = lift_zd_to_dim(a16, b16, 16, (1, 2), (3, 4), &sc);
        let total_probes = ch.union_probes.len();
        let optimal_probes = ch.optimal_probes.len();
        println!(
            "16D ZD pair: {} union probes, {} optimal probes (of 16 basis elements)",
            total_probes, optimal_probes
        );
        assert!(
            total_probes > 0,
            "ZD channel should have at least one valid probe direction"
        );
    }

    #[test]
    fn test_chsh_runs_at_16d() {
        let result = chsh_violation_test(16, 1000, 42);
        assert_eq!(result.dimension, 16);
        assert!(result.n_zd_pairs > 0);
        println!(
            "16D CHSH: S = {:.4} +/- {:.4}, violates = {}",
            result.s_value, result.s_error, result.violates_classical
        );
        println!(
            "  ZD pairs = {}, usable = {}, optimal_probes = {}, shared_paths = {}",
            result.n_zd_pairs,
            result.n_usable_channels,
            result.n_optimal_probes,
            result.n_shared_paths
        );
        println!("  Correlators: {:?}", result.correlators);
    }

    #[test]
    fn test_chsh_runs_at_512d() {
        let result = chsh_violation_test(512, 1000, 42);
        assert_eq!(result.dimension, 512);
        println!(
            "512D CHSH: S = {:.4} +/- {:.4}, violates = {}",
            result.s_value, result.s_error, result.violates_classical
        );
        println!(
            "  ZD pairs = {}, usable = {}, optimal_probes = {}, shared_paths = {}",
            result.n_zd_pairs,
            result.n_usable_channels,
            result.n_optimal_probes,
            result.n_shared_paths
        );
        println!("  Correlators: {:?}", result.correlators);
    }

    #[test]
    fn test_shared_zd_paths_nonempty() {
        let paths = find_shared_zd_paths(16);
        assert!(
            !paths.is_empty(),
            "Should find at least one connected component"
        );
        let total_nodes: usize = paths.iter().map(|c| c.len()).sum();
        println!(
            "16D ZD paths: {} components, {} total nodes",
            paths.len(),
            total_nodes
        );
    }

    #[test]
    fn test_dimensional_scaling() {
        let results = chsh_dimensional_scaling(64, 500, 42);
        assert!(results.len() >= 2);
        for (dim, result) in &results {
            println!(
                "dim={}: S = {:.4} +/- {:.4}, violates = {}, optimal_probes = {}",
                dim,
                result.s_value,
                result.s_error,
                result.violates_classical,
                result.n_optimal_probes,
            );
        }
    }
}
