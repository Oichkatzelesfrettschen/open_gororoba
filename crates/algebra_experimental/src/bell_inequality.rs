use cd_kernel::cayley_dickson::{cd_associator, cd_multiply, cd_norm_sq};
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
struct ZdChannel {
    a: Vec<f64>,
    b: Vec<f64>,
    /// Basis indices with non-zero support in a or b
    basis_indices: Vec<usize>,
    /// Optimal probes: basis elements where the associator [A', e_k, B'] is non-zero
    /// across ALL four (Alice plane, Bob plane) basis combinations.
    /// These are X_optimal = X_{a1,b1} AND X_{a1,b2} AND X_{a2,b1} AND X_{a2,b2}.
    optimal_probes: Vec<usize>,
    /// Union probes: basis elements where at least one [e_ai, e_k, e_bj] is non-zero.
    /// Superset of optimal_probes; used as fallback when optimal set is empty.
    union_probes: Vec<usize>,
}

/// Lift a sedenion (16D) 2-blade ZD pair to target dimension via Cayley-Dickson doubling.
///
/// Embedding: A_512 = (A_16, 0, 0, ..., 0). Because the right-hand elements are zero,
/// conjugate terms in the doubling formula drop out, and A*B = 0 is strictly preserved.
fn lift_zd_to_dim(
    a_16: &[f64; 16],
    b_16: &[f64; 16],
    target_dim: usize,
    alice_plane: (usize, usize),
    bob_plane: (usize, usize),
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

    // Compute probe validity for all four (Alice basis, Bob basis) combinations.
    // For Alice rotating in plane (a1, a2), her rotated element is:
    //   A' = cos(theta_a) * e_{a1} + sin(theta_a) * e_{a2}  (within her support)
    // Similarly for Bob in plane (b1, b2).
    //
    // The associator is trilinear, so:
    //   [A', X, B'] = sum over (ai, bj) of cos/sin * [e_ai, e_k, e_bj]
    //
    // For non-trivial correlations at ALL angles, X must be valid for ALL four combos.
    let (a1, a2) = alice_plane;
    let (b1, b2) = bob_plane;

    let probes_a1_b1 = find_valid_probes_basis(&a, &b, a1, b1, target_dim);
    let probes_a1_b2 = find_valid_probes_basis(&a, &b, a1, b2, target_dim);
    let probes_a2_b1 = find_valid_probes_basis(&a, &b, a2, b1, target_dim);
    let probes_a2_b2 = find_valid_probes_basis(&a, &b, a2, b2, target_dim);

    // X_optimal = intersection of all four sets
    let optimal_probes: Vec<usize> = probes_a1_b1
        .iter()
        .copied()
        .filter(|k| {
            probes_a1_b2.contains(k)
                && probes_a2_b1.contains(k)
                && probes_a2_b2.contains(k)
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
        basis_indices,
        optimal_probes,
        union_probes,
    }
}

/// Find basis elements e_k where [A_rotated_in_axis_i, e_k, B_rotated_in_axis_j] != 0.
///
/// This tests the associator with A perturbed along axis `alice_axis` and B along `bob_axis`,
/// checking which probe directions k produce non-zero torque.
///
/// Optimization: for sedenion-lifted ZD pairs (support in indices 0..15), the XOR-based
/// index mapping confines products to the same power-of-2 block. We compute the actual
/// support bound and limit the scan to 2x that range to capture cross terms.
fn find_valid_probes_basis(
    a: &[f64],
    b: &[f64],
    alice_axis: usize,
    bob_axis: usize,
    dim: usize,
) -> Vec<usize> {
    let theta_test = 0.3;
    let a_perturbed = rotate_in_plane(a, alice_axis, next_axis(alice_axis, dim), theta_test);
    let b_perturbed = rotate_in_plane(b, bob_axis, next_axis(bob_axis, dim), theta_test);

    // Compute support bound: highest non-zero index in the perturbed elements.
    // For sedenion-lifted pairs, this is typically 15 (or includes rotation axes).
    let max_a = a_perturbed
        .iter()
        .enumerate()
        .rev()
        .find(|&(_, v)| v.abs() > 1e-15)
        .map(|(i, _)| i + 1)
        .unwrap_or(0);
    let max_b = b_perturbed
        .iter()
        .enumerate()
        .rev()
        .find(|&(_, v)| v.abs() > 1e-15)
        .map(|(i, _)| i + 1)
        .unwrap_or(0);
    let max_support = max_a.max(max_b);

    // XOR products of indices < N stay < N when N is a power of 2.
    // Scan up to 2 * next_power_of_two(max_support) to be safe, capped by dim.
    let scan_limit = dim.min(max_support.next_power_of_two() * 2);

    (0..scan_limit)
        .filter(|&k| {
            let mut x = vec![0.0; dim];
            x[k] = 1.0;
            let assoc = cd_associator(&a_perturbed, &x, &b_perturbed);
            let norm_sq: f64 = assoc.iter().map(|v| v * v).sum();
            norm_sq > 1e-20
        })
        .collect()
}

/// Get a companion axis for rotation: the other axis in the pair.
fn next_axis(axis: usize, dim: usize) -> usize {
    if axis + 1 < dim { axis + 1 } else { 1 }
}

/// Known sedenion 2-blade zero-divisor pairs from the box-kite structure.
///
/// In sedenions (dim=16), the box-kite construction (de Marrais 2001) produces
/// ZD pairs. Each is a 2-blade: (e_i + e_j) * (e_k +/- e_l) = 0.
///
/// Generated by brute-force enumeration at dim=16 (only ~16K checks).
fn known_sedenion_zd_pairs() -> Vec<([f64; 16], [f64; 16])> {
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
fn rotate_in_plane(x: &[f64], axis_a: usize, axis_b: usize, theta: f64) -> Vec<f64> {
    let mut result = x.to_vec();
    let (c, s) = (theta.cos(), theta.sin());
    let xa = x[axis_a];
    let xb = x[axis_b];
    result[axis_a] = xa * c - xb * s;
    result[axis_b] = xa * s + xb * c;
    result
}

/// Compute the associator-based measurement outcome.
///
/// Given a ZD channel (A, B), measurement angle theta for the measured party,
/// rotation plane, and probe element e_k:
///   outcome = sign( sum_j [A'(theta), e_k, B]_j )
///
/// The probe e_k must be valid (non-zero associator with both A and B sides).
fn associator_measurement(
    measured: &[f64],
    partner: &[f64],
    theta: f64,
    plane: (usize, usize),
    probe_idx: usize,
    dim: usize,
) -> f64 {
    let rotated = rotate_in_plane(measured, plane.0, plane.1, theta);

    let mut x = vec![0.0; dim];
    x[probe_idx] = 1.0;

    let assoc = cd_associator(&rotated, &x, partner);
    let total: f64 = assoc.iter().sum();

    if total >= 0.0 { 1.0 } else { -1.0 }
}

/// CHSH measurement configuration.
struct ChshConfig {
    alice_plane: (usize, usize),
    bob_plane: (usize, usize),
    dim: usize,
    n_trials: usize,
}

/// Compute the CHSH correlator E(theta_a, theta_b).
///
/// For each trial:
/// 1. Pick a random ZD channel (shared entanglement resource)
/// 2. Pick a random *optimal* probe (valid across all angle combos; falls back to union)
/// 3. Alice: sign([A'(theta_a), e_k, B])
/// 4. Bob: sign([B'(theta_b), e_k, A])
/// 5. Record product of outcomes
///
/// E = <outcome_a * outcome_b> averaged over n_trials.
fn compute_chsh_correlator(
    channels: &[ZdChannel],
    theta_a: f64,
    theta_b: f64,
    cfg: &ChshConfig,
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
        let ch = usable[rng.gen_range(0..usable.len())];

        // Prefer optimal probes; fall back to union if optimal is empty
        let probes = if !ch.optimal_probes.is_empty() {
            &ch.optimal_probes
        } else {
            &ch.union_probes
        };
        let probe = probes[rng.gen_range(0..probes.len())];

        // Alice measures: sign([A'(theta_a), e_probe, B])
        let outcome_a =
            associator_measurement(&ch.a, &ch.b, theta_a, cfg.alice_plane, probe, cfg.dim);

        // Bob measures: sign([B'(theta_b), e_probe, A])
        let outcome_b =
            associator_measurement(&ch.b, &ch.a, theta_b, cfg.bob_plane, probe, cfg.dim);

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

    // Step 1: Get sedenion ZD pairs and lift to target dimension
    let sed_pairs = known_sedenion_zd_pairs();
    let channels: Vec<ZdChannel> = sed_pairs
        .iter()
        .map(|(a, b)| lift_zd_to_dim(a, b, dim, alice_plane, bob_plane))
        .collect();

    let n_zd_pairs = channels.len();
    let n_usable = channels
        .iter()
        .filter(|ch| !ch.optimal_probes.is_empty() || !ch.union_probes.is_empty())
        .count();
    let n_optimal_probes: usize = channels.iter().map(|ch| ch.optimal_probes.len()).sum();

    // Step 2: CHSH measurement settings (optimal for max quantum violation)
    let a = 0.0;
    let a_prime = std::f64::consts::FRAC_PI_2;
    let b = std::f64::consts::FRAC_PI_4;
    let b_prime = -std::f64::consts::FRAC_PI_4;

    // Step 3: Compute all four correlators
    let cfg = ChshConfig {
        alice_plane,
        bob_plane,
        dim,
        n_trials: n_samples,
    };

    let e_ab = compute_chsh_correlator(&channels, a, b, &cfg, &mut rng);
    let e_ab_prime = compute_chsh_correlator(&channels, a, b_prime, &cfg, &mut rng);
    let e_a_prime_b = compute_chsh_correlator(&channels, a_prime, b, &cfg, &mut rng);
    let e_a_prime_b_prime = compute_chsh_correlator(&channels, a_prime, b_prime, &cfg, &mut rng);

    // Step 4: S-value
    let s_value = e_ab - e_ab_prime + e_a_prime_b + e_a_prime_b_prime;

    // Step 5: Statistical error (SE for binary outcomes)
    let s_error = 4.0 / (n_samples as f64).sqrt();

    // Step 6: Shared ZD paths
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

/// Count shared ZD paths: pairs of channels with overlapping basis support.
fn count_shared_zd_paths(channels: &[ZdChannel]) -> usize {
    let mut count = 0;
    for (i, ch_i) in channels.iter().enumerate() {
        let set_i: HashSet<usize> = ch_i.basis_indices.iter().copied().collect();
        for ch_j in channels.iter().skip(i + 1) {
            let set_j: HashSet<usize> = ch_j.basis_indices.iter().copied().collect();
            if set_i.intersection(&set_j).next().is_some() {
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

    let sed_pairs = known_sedenion_zd_pairs();
    let channels: Vec<ZdChannel> = sed_pairs
        .iter()
        .map(|(a, b)| lift_zd_to_dim(a, b, dim, alice_plane, bob_plane))
        .collect();

    let n = channels.len();
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];

    for (i, ch_i) in channels.iter().enumerate() {
        let set_i: HashSet<usize> = ch_i.basis_indices.iter().copied().collect();
        for (j, ch_j) in channels.iter().enumerate().skip(i + 1) {
            let set_j: HashSet<usize> = ch_j.basis_indices.iter().copied().collect();
            if set_i.intersection(&set_j).next().is_some() {
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
                i, norm
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
            let ch = lift_zd_to_dim(a16, b16, target_dim, alice_plane, bob_plane);
            let ab = cd_multiply(&ch.a, &ch.b);
            let norm = cd_norm_sq(&ab).sqrt();
            assert!(
                norm < 1e-10,
                "Lifted ZD at dim={} has ||a*b|| = {}, expected ~0",
                target_dim, norm
            );
        }
    }

    #[test]
    fn test_valid_probes_exist() {
        let pairs = known_sedenion_zd_pairs();
        assert!(!pairs.is_empty());

        let (a16, b16) = &pairs[0];
        let ch = lift_zd_to_dim(a16, b16, 16, (1, 2), (3, 4));
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
                dim, result.s_value, result.s_error, result.violates_classical,
                result.n_optimal_probes,
            );
        }
    }
}
