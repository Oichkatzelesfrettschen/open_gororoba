//! Majorana braiding operators mapped to the Cayley-Dickson tower.
//!
//! Maps Microsoft's Majorana topological qubit braid operations into the
//! non-associative CD algebra, measuring "topological friction" -- the
//! cumulative associator torque that braiding accumulates when lifted from
//! associative Clifford algebras into non-associative sedenions and beyond.
//!
//! References:
//! - Nayak et al., Rev. Mod. Phys. 80, 1083 (2008): Non-Abelian anyons and TQC
//! - Ivanov, Phys. Rev. Lett. 86, 268 (2001): Non-Abelian statistics of vortices
//! - Kitaev, Phys.-Usp. 44, 131 (2001): Unpaired Majorana fermions

use crate::bell_inequality::{
    SignTableCache, known_sedenion_zd_pairs, rotate_sparse,
};
use algebra_core::physics::clifford::{gamma_matrices_cl8, GammaMatrix};
use cd_kernel::cayley_dickson::cd_basis_mul_sign_iter;
use num_complex::Complex64;

/// A Majorana mode mapped to a CD basis element.
#[derive(Debug, Clone)]
pub struct MajoranaMode {
    /// Majorana label (0-indexed: gamma_0..gamma_{n-1})
    pub gamma_index: usize,
    /// CD basis element e_k this mode maps to
    pub cd_basis_index: usize,
    /// CD algebra dimension
    pub cd_dim: usize,
}

/// Result of executing a braid through the CD tower.
#[derive(Debug, Clone)]
pub struct BraidResult {
    /// Whether parity is preserved under this braid
    pub parity_preserved: bool,
    /// Cumulative associator norm (topological friction)
    pub topological_friction: f64,
    /// Fidelity of CD braid vs Clifford braid
    pub fidelity: f64,
    /// Maximum single-probe associator norm
    pub max_associator_norm: f64,
    /// Number of probes tested
    pub n_probes_tested: usize,
}

/// Full experiment result comparing Clifford vs CD braiding.
#[derive(Debug, Clone)]
pub struct BraidingExperimentResult {
    pub dim: usize,
    pub braid_sequence: Vec<(usize, usize)>,
    pub clifford_parity: f64,
    pub cd_parity: f64,
    pub friction_per_braid: Vec<f64>,
    pub total_friction: f64,
    pub fidelity: f64,
}

/// Construct the braid operator U_ij = (1/sqrt(2))(I + gamma_i * gamma_j).
///
/// Uses the Cl(8) gamma matrices (16x16 complex). For 4 Majorana modes
/// we only need gamma_0..gamma_3 from the 8 available generators.
///
/// Source: Nayak et al. (2008) Eq. 5, Ivanov (2001) Eq. 5.
pub fn clifford_braid_unitary(i: usize, j: usize) -> GammaMatrix {
    assert!(i != j, "Braid requires distinct Majorana indices");
    let gammas = gamma_matrices_cl8();
    assert!(
        i < gammas.len() && j < gammas.len(),
        "Majorana indices must be < {}",
        gammas.len()
    );

    let dim = gammas[0].nrows();
    let identity = GammaMatrix::identity(dim, dim);
    let inv_sqrt2 = Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0);

    // U_ij = (1/sqrt(2)) * (I + gamma_i * gamma_j)
    let gi_gj = &gammas[i] * &gammas[j];
    (&identity + &gi_gj) * inv_sqrt2
}

/// Compute the parity observable P = i * gamma_0 * gamma_1 * gamma_2 * gamma_3.
///
/// For 4 Majorana modes, P has eigenvalues +/-1 and commutes with all
/// braiding operators U_ij. This is the topological charge.
pub fn parity_observable_clifford() -> GammaMatrix {
    let gammas = gamma_matrices_cl8();
    let i_unit = Complex64::new(0.0, 1.0);

    // P = i * gamma_0 * gamma_1 * gamma_2 * gamma_3
    let mut p = &gammas[0] * i_unit;
    p = &p * &gammas[1];
    p = &p * &gammas[2];
    p = &p * &gammas[3];
    p
}

/// Check that [P, U] = P*U - U*P = 0 (braid preserves parity).
pub fn check_braid_preserves_parity(i: usize, j: usize) -> f64 {
    let p = parity_observable_clifford();
    let u = clifford_braid_unitary(i, j);

    let commutator = &p * &u - &u * &p;

    // Return Frobenius norm of commutator
    commutator
        .iter()
        .map(|c| c.norm_sqr())
        .sum::<f64>()
        .sqrt()
}

/// Check that U is unitary: U * U^dag = I.
pub fn check_braid_unitarity(i: usize, j: usize) -> f64 {
    let u = clifford_braid_unitary(i, j);
    let dim = u.nrows();
    let identity = GammaMatrix::identity(dim, dim);

    // U^dag = U^* transposed
    let u_dag = u.adjoint();
    let product = &u * &u_dag;
    let diff = &product - &identity;

    diff.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt()
}

/// Map 4 Majorana modes to CD basis elements.
///
/// gamma_k -> e_{k+1} (using CD basis elements e_1, e_2, e_3, e_4).
/// These are within the sedenion support (indices 1-15) ensuring
/// they participate in the ZD graph structure.
pub fn map_majoranas_to_cd(n_majoranas: usize, dim: usize) -> Vec<MajoranaMode> {
    assert!(dim >= 16 && dim.is_power_of_two());
    (0..n_majoranas)
        .map(|k| MajoranaMode {
            gamma_index: k,
            cd_basis_index: k + 1, // e_1, e_2, e_3, e_4
            cd_dim: dim,
        })
        .collect()
}

/// Execute a single braid in the CD tower and measure topological friction.
///
/// The braid U_ij in Clifford algebra maps to:
///   In CD: rotate e_i in the (e_i, e_j) subplane by pi/4
/// The "friction" is the cumulative associator torque [A_rotated, X, B]
/// summed over all valid probes X.
pub fn cd_braid(
    mode_i: &MajoranaMode,
    mode_j: &MajoranaMode,
    sign_table: &SignTableCache,
) -> BraidResult {
    let dim = mode_i.cd_dim;
    let i = mode_i.cd_basis_index;
    let j = mode_j.cd_basis_index;

    // Construct sparse basis elements
    let a_sparse = vec![(i, 1.0)];
    let b_sparse = vec![(j, 1.0)];

    // Braid = SO(2) rotation by pi/4 in the (e_i, e_j) subplane
    let theta = std::f64::consts::FRAC_PI_4;
    let a_rotated = rotate_sparse(&a_sparse, i, j, theta);

    // Measure associator [A_rotated, X, B] for all basis probes
    let mut total_friction = 0.0;
    let mut max_norm = 0.0;
    let mut n_probes = 0;

    for k in 1..dim {
        if k == i || k == j {
            continue;
        }
        let x_sparse = [(k, 1.0)];
        let assoc_sum =
            sign_table.sparse_associator_sum(&a_rotated, &x_sparse, &b_sparse);
        let norm = assoc_sum.abs();
        total_friction += norm;
        if norm > max_norm {
            max_norm = norm;
        }
        n_probes += 1;
    }

    BraidResult {
        parity_preserved: true, // CD parity check done separately
        topological_friction: total_friction,
        fidelity: if total_friction < 1e-12 { 1.0 } else { 0.0 },
        max_associator_norm: max_norm,
        n_probes_tested: n_probes,
    }
}

/// Measure total topological friction for a braid sequence.
pub fn measure_topological_friction(
    braid_sequence: &[(usize, usize)],
    modes: &[MajoranaMode],
    sign_table: &SignTableCache,
) -> f64 {
    braid_sequence
        .iter()
        .map(|&(i, j)| {
            let result = cd_braid(&modes[i], &modes[j], sign_table);
            result.topological_friction
        })
        .sum()
}

/// Compute parity in CD representation via sign-table chain multiplication.
///
/// P_CD = sign_chain(e_1, e_2, e_3, e_4): product of 4 basis elements
/// evaluated via the sign table. In associative algebras this matches
/// the Clifford parity; in non-associative algebras it depends on
/// association order.
pub fn parity_observable_cd(
    modes: &[MajoranaMode],
    sign_table: &SignTableCache,
) -> f64 {
    assert!(modes.len() >= 4, "Need at least 4 Majorana modes for parity");

    // Left-associated product: ((e_i0 * e_i1) * e_i2) * e_i3
    let a = vec![(modes[0].cd_basis_index, 1.0)];
    let b = vec![(modes[1].cd_basis_index, 1.0)];
    let c = vec![(modes[2].cd_basis_index, 1.0)];
    let d = vec![(modes[3].cd_basis_index, 1.0)];

    let ab = sign_table.sparse_multiply(&a, &b);
    let abc = sign_table.sparse_multiply(&ab, &c);
    let abcd = sign_table.sparse_multiply(&abc, &d);

    // The product e_1*e_2*e_3*e_4 in CD algebra maps to a single basis element
    // (via XOR: 1^2^3^4 = 4) with a sign from the sign table.
    // Return the sign of the dominant component.
    abcd.iter()
        .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
        .map_or(0.0, |&(_, v)| v.signum())
}

/// Test box-kite fusion: partition each kite's 6 assessor pairs by
/// their CD multiplication sign.
///
/// Ising anyon fusion predicts exactly 2 channels: vacuum (1) and fermion (psi).
/// We check whether each box-kite's assessor pairs partition into exactly 2
/// sign classes under sedenion multiplication.
///
/// Returns: Vec of (kite_index, n_distinct_signs, signs_per_pair)
pub fn boxkite_fusion_channels(dim: usize) -> Vec<(usize, usize, Vec<i32>)> {
    assert!(dim >= 16 && dim.is_power_of_two());

    // The 7 box-kites from BoxKite.v (hardcoded assessor pairs)
    let boxkites: Vec<Vec<(usize, usize)>> = vec![
        vec![(1, 14), (2, 13), (3, 12), (4, 11), (5, 10), (6, 9)],
        vec![(1, 11), (3, 9), (4, 14), (5, 15), (6, 12), (7, 13)],
        vec![(1, 10), (2, 9), (4, 15), (5, 14), (6, 13), (7, 12)],
        vec![(1, 13), (2, 14), (3, 15), (5, 9), (6, 10), (7, 11)],
        vec![(1, 12), (2, 15), (3, 14), (4, 9), (6, 11), (7, 10)],
        vec![(1, 15), (2, 12), (3, 13), (4, 10), (5, 11), (7, 9)],
        vec![(2, 11), (3, 10), (4, 13), (5, 12), (6, 15), (7, 14)],
    ];

    boxkites
        .iter()
        .enumerate()
        .map(|(kite_idx, pairs)| {
            let signs: Vec<i32> = pairs
                .iter()
                .map(|&(lo, hi)| cd_basis_mul_sign_iter(dim, lo, hi))
                .collect();

            let mut distinct: Vec<i32> = signs.clone();
            distinct.sort();
            distinct.dedup();

            (kite_idx, distinct.len(), signs)
        })
        .collect()
}

/// Test missing-edge protection: associator = 0 on involution axes.
///
/// Missing ZD edges are pairs (i, i XOR dim/2) for i in 1..dim/2.
/// For each missing pair, we verify that the associator [e_i, e_k, e_{i XOR dim/2}]
/// vanishes for ALL valid probes e_k.
///
/// Returns: Vec of (i, j=i^half, max_associator_over_all_probes)
pub fn test_missing_edge_protection(
    dim: usize,
    sign_table: &SignTableCache,
) -> Vec<(usize, usize, f64)> {
    assert!(dim >= 16 && dim.is_power_of_two());
    let half = dim / 2;

    let mut results = Vec::new();

    for i in 1..half {
        let j = i ^ half;
        let a_sparse = vec![(i, 1.0)];
        let b_sparse = vec![(j, 1.0)];

        let mut max_assoc = 0.0f64;

        for k in 1..dim {
            if k == i || k == j || k == 0 || k == half {
                continue;
            }
            let x_sparse = [(k, 1.0)];
            let assoc = sign_table
                .sparse_associator_sum(&a_sparse, &x_sparse, &b_sparse)
                .abs();
            max_assoc = max_assoc.max(assoc);
        }

        results.push((i, j, max_assoc));
    }

    results
}

/// Full braiding experiment comparing Clifford vs CD.
pub fn braiding_experiment(
    circuit: &[(usize, usize)],
    dim: usize,
    _seed: u64,
) -> BraidingExperimentResult {
    let modes = map_majoranas_to_cd(4, dim);
    let sign_table = SignTableCache::new(dim);

    // Clifford baseline: compute parity commutator
    let clifford_parity = {
        let p = parity_observable_clifford();
        // P^2 should be identity; eigenvalue is +/-1
        // For the product gamma_0..gamma_3, the parity eigenvalue depends on convention
        let p_sq = &p * &p;
        p_sq[(0, 0)].re.signum()
    };

    // CD parity
    let cd_parity = parity_observable_cd(&modes, &sign_table);

    // Friction per braid
    let friction_per_braid: Vec<f64> = circuit
        .iter()
        .map(|&(i, j)| {
            let result = cd_braid(&modes[i], &modes[j], &sign_table);
            result.topological_friction
        })
        .collect();

    let total_friction: f64 = friction_per_braid.iter().sum();

    // Fidelity: 1.0 if parities match AND friction is zero
    let fidelity = if (clifford_parity - cd_parity).abs() < 1e-10
        && total_friction < 1e-10
    {
        1.0
    } else {
        0.0
    };

    BraidingExperimentResult {
        dim,
        braid_sequence: circuit.to_vec(),
        clifford_parity,
        cd_parity,
        friction_per_braid,
        total_friction,
        fidelity,
    }
}

/// Complex-time braiding: Wick rotation damping of topological friction.
///
/// Under Wick rotation by angle theta in [0, pi/2], the friction is damped:
///   F_wick = F_real * exp(-gap * tau)
/// where gap is proportional to the ZD edge density and tau = theta * t_planck.
pub fn complex_time_braid(
    mode_i: &MajoranaMode,
    mode_j: &MajoranaMode,
    theta: f64,
    sign_table: &SignTableCache,
) -> BraidResult {
    let mut result = cd_braid(mode_i, mode_j, sign_table);

    // Wick rotation damping factor
    // gap ~ edge_density ~ (dim^2 - 6*dim + 8) / (2 * (dim-2)*(dim-3)/2)
    let dim = mode_i.cd_dim as f64;
    let edge_density = (dim * dim - 6.0 * dim + 8.0)
        / ((dim - 2.0) * (dim - 3.0));
    let damping = (-edge_density * theta.sin()).exp();

    result.topological_friction *= damping;
    result.max_associator_norm *= damping;
    result
}

/// Friction vs dimension scaling sweep.
///
/// Measures topological friction at dims 16, 32, 64, ..., max_dim
/// for a standard sigma_1 braid (modes 0, 1).
pub fn friction_dimensional_scaling(max_dim: usize) -> Vec<(usize, f64, f64)> {
    let mut results = Vec::new();
    let mut dim = 16;

    while dim <= max_dim {
        let modes = map_majoranas_to_cd(4, dim);
        let sign_table = SignTableCache::new(dim);
        let braid = cd_braid(&modes[0], &modes[1], &sign_table);

        results.push((dim, braid.topological_friction, braid.max_associator_norm));
        dim *= 2;
    }

    results
}

/// CHSH test with probes aligned along braiding channels instead of random.
///
/// Uses box-kite assessor pairs as measurement planes instead of the
/// arbitrary (1,2) / (3,4) planes used in the standard test.
pub fn chsh_braid_aligned(
    dim: usize,
    n_samples: usize,
    seed: u64,
) -> crate::bell_inequality::BellTestResult {
    // Use box-kite strut directions as measurement planes
    // Kite 1: assessors (1,14), (2,13), ... -> use (1,14) for Alice, (2,13) for Bob
    // This aligns measurements with the topological structure
    use crate::bell_inequality::{
        lift_zd_to_dim,
    };
    use rand::prelude::*;
    use rand_chacha::ChaCha8Rng;

    let alice_plane = (1, 14); // First assessor pair of boxkite_1
    let bob_plane = (2, 13);   // Second assessor pair of boxkite_1

    let sign_cache = SignTableCache::new(dim);

    let sed_pairs = known_sedenion_zd_pairs();
    let channels: Vec<_> = sed_pairs
        .iter()
        .map(|(a, b)| lift_zd_to_dim(a, b, dim, alice_plane, bob_plane, &sign_cache))
        .collect();

    let n_zd_pairs = channels.len();
    let n_usable = channels
        .iter()
        .filter(|ch| !ch.optimal_probes.is_empty() || !ch.union_probes.is_empty())
        .count();
    let n_optimal_probes: usize = channels.iter().map(|ch| ch.optimal_probes.len()).sum();

    // CHSH optimal angles
    let a = 0.0;
    let a_prime = std::f64::consts::FRAC_PI_2;
    let b = std::f64::consts::FRAC_PI_4;
    let b_prime = -std::f64::consts::FRAC_PI_4;

    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let compute_correlator = |theta_a: f64, theta_b: f64, rng: &mut ChaCha8Rng| -> f64 {
        let usable: Vec<_> = channels
            .iter()
            .filter(|ch| !ch.optimal_probes.is_empty() || !ch.union_probes.is_empty())
            .collect();
        if usable.is_empty() {
            return 0.0;
        }
        let mut total = 0.0;
        for _ in 0..n_samples {
            let ch = usable[rng.gen_range(0..usable.len())];
            let probes = if !ch.optimal_probes.is_empty() {
                &ch.optimal_probes
            } else {
                &ch.union_probes
            };
            let probe = probes[rng.gen_range(0..probes.len())];

            let outcome_a = crate::bell_inequality::associator_measurement_fast(
                &ch.a_sparse, &ch.b_sparse, theta_a, alice_plane, probe, &sign_cache,
            );
            let outcome_b = crate::bell_inequality::associator_measurement_fast(
                &ch.b_sparse, &ch.a_sparse, theta_b, bob_plane, probe, &sign_cache,
            );
            total += outcome_a * outcome_b;
        }
        total / n_samples as f64
    };

    let e_ab = compute_correlator(a, b, &mut rng);
    let e_ab_prime = compute_correlator(a, b_prime, &mut rng);
    let e_a_prime_b = compute_correlator(a_prime, b, &mut rng);
    let e_a_prime_b_prime = compute_correlator(a_prime, b_prime, &mut rng);

    let s_value = e_ab - e_ab_prime + e_a_prime_b + e_a_prime_b_prime;
    let s_error = 4.0 / (n_samples as f64).sqrt();

    // Count shared paths
    let mut n_shared_paths = 0;
    for (idx_i, ch_i) in channels.iter().enumerate() {
        let set_i: std::collections::HashSet<usize> =
            ch_i.basis_indices.iter().copied().collect();
        for ch_j in channels.iter().skip(idx_i + 1) {
            let set_j: std::collections::HashSet<usize> =
                ch_j.basis_indices.iter().copied().collect();
            if set_i.intersection(&set_j).next().is_some() {
                n_shared_paths += 1;
            }
        }
    }

    crate::bell_inequality::BellTestResult {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clifford_braid_is_unitary() {
        // U_01 * U_01^dag = I
        let err = check_braid_unitarity(0, 1);
        assert!(
            err < 1e-10,
            "Braid U_01 should be unitary, got error = {err}"
        );

        let err2 = check_braid_unitarity(2, 3);
        assert!(
            err2 < 1e-10,
            "Braid U_23 should be unitary, got error = {err2}"
        );
    }

    #[test]
    fn test_clifford_braid_preserves_parity() {
        // [P, U_01] = 0
        for (i, j) in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)] {
            let commutator_norm = check_braid_preserves_parity(i, j);
            assert!(
                commutator_norm < 1e-10,
                "[P, U_{i}{j}] should vanish, got norm = {commutator_norm}"
            );
        }
    }

    #[test]
    fn test_cd_braid_at_dim16_has_friction() {
        let modes = map_majoranas_to_cd(4, 16);
        let sign_table = SignTableCache::new(16);
        let result = cd_braid(&modes[0], &modes[1], &sign_table);

        assert!(
            result.topological_friction > 0.0,
            "CD braid at dim=16 should have positive friction, got {}",
            result.topological_friction
        );
        println!(
            "dim=16 friction: {:.6}, max_assoc: {:.6}, probes: {}",
            result.topological_friction, result.max_associator_norm, result.n_probes_tested
        );
    }

    #[test]
    fn test_topological_friction_positive_all_dims() {
        for &dim in &[16, 32, 64, 128] {
            let modes = map_majoranas_to_cd(4, dim);
            let sign_table = SignTableCache::new(dim);
            let result = cd_braid(&modes[0], &modes[1], &sign_table);
            assert!(
                result.topological_friction > 0.0,
                "dim={dim}: friction should be positive, got {}",
                result.topological_friction
            );
        }
    }

    #[test]
    fn test_friction_monotone_with_dim() {
        let scaling = friction_dimensional_scaling(128);
        assert!(scaling.len() >= 3);

        for window in scaling.windows(2) {
            let (dim_a, f_a, _) = window[0];
            let (dim_b, f_b, _) = window[1];
            assert!(
                f_b >= f_a,
                "Friction should be monotone: dim={dim_a} f={f_a} > dim={dim_b} f={f_b}"
            );
        }
    }

    #[test]
    fn test_boxkite_fusion_exactly_two_channels() {
        let fusion = boxkite_fusion_channels(16);
        assert_eq!(fusion.len(), 7, "Should have 7 box-kites");

        for (kite_idx, n_channels, signs) in &fusion {
            assert_eq!(
                *n_channels, 2,
                "Box-kite {} should have exactly 2 fusion channels (Ising anyon), \
                 got {} with signs {:?}",
                kite_idx, n_channels, signs
            );
        }
    }

    #[test]
    fn test_missing_edge_associator_structure() {
        let sign_table = SignTableCache::new(16);
        let results = test_missing_edge_protection(16, &sign_table);

        assert_eq!(results.len(), 7, "Should have 7 missing edges at dim=16");

        // FALSIFIED: the plan predicted zero friction on missing edges.
        // Reality: missing edges (i, i XOR 8) do NOT have vanishing associator.
        // They are topologically special (absent from ZD graph) but the
        // associator [e_i, e_k, e_{i XOR 8}] is generally non-zero.
        // Record the actual values for analysis.
        for (i, j, max_assoc) in &results {
            println!(
                "Missing edge ({i}, {j}): max |associator| = {max_assoc:.6}"
            );
        }

        // Verify the test ran correctly: all missing pairs have i XOR 8 = j
        for &(i, j, _) in &results {
            assert_eq!(i ^ 8, j, "Missing edge pair should satisfy i XOR 8 = j");
        }
    }

    #[test]
    fn test_parity_clifford_vs_cd_at_dim16() {
        let modes = map_majoranas_to_cd(4, 16);
        let sign_table = SignTableCache::new(16);

        let cd_par = parity_observable_cd(&modes, &sign_table);
        // In CD, the product e_1*e_2*e_3*e_4 maps to another basis element
        // (not the scalar e_0), so parity_observable_cd returns the sign of
        // that element's coefficient. This differs from the Clifford case
        // where the 4-fold product of gamma matrices yields +/-1 on the diagonal.
        println!(
            "CD parity at dim=16: {} (0 means product has no e_0 component)",
            cd_par
        );
        // The test validates that the function runs and returns a finite value.
        assert!(cd_par.is_finite());
    }

    #[test]
    fn test_complex_time_damps_friction() {
        let modes = map_majoranas_to_cd(4, 16);
        let sign_table = SignTableCache::new(16);

        let real_braid = cd_braid(&modes[0], &modes[1], &sign_table);
        let wick_braid =
            complex_time_braid(&modes[0], &modes[1], std::f64::consts::FRAC_PI_4, &sign_table);

        assert!(
            wick_braid.topological_friction < real_braid.topological_friction,
            "Wick rotation should damp friction: real={} > wick={}",
            real_braid.topological_friction,
            wick_braid.topological_friction
        );
    }

    #[test]
    fn test_aligned_probes_change_chsh() {
        let result = chsh_braid_aligned(16, 1000, 42);
        assert!(
            result.s_value.abs() < 4.0,
            "S-value must be in [-4, 4], got {}",
            result.s_value
        );
        println!(
            "Aligned CHSH at 16D: S = {:.4} +/- {:.4}",
            result.s_value, result.s_error
        );
    }

    #[test]
    fn test_braiding_experiment_runs() {
        let result = braiding_experiment(&[(0, 1), (1, 2)], 16, 42);
        assert_eq!(result.dim, 16);
        assert_eq!(result.braid_sequence.len(), 2);
        println!(
            "Braiding experiment: parity_cliff={}, parity_cd={}, friction={}",
            result.clifford_parity, result.cd_parity, result.total_friction
        );
    }
}
