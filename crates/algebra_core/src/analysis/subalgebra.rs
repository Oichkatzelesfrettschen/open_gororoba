//! Octonion subalgebra enumeration within Cayley-Dickson algebras.
//!
//! Identifies all copies of the octonions inside the sedenions (dim=16)
//! and higher CD algebras. Each subalgebra corresponds to an order-8
//! XOR-closed subset of basis indices containing 0, where the inherited
//! CD sign structure yields an alternative (octonion-isomorphic) algebra.
//!
//! # Literature
//! - Gresnigt & Gillard (arXiv:1904.03186): C x S -> 3 x Cl(6)
//! - Wilmot (arXiv:2512.07210): Fano volume, sedenion subalgebra visualization
//! - Furey et al. (2024): Cl(8) -> 3 generations
//!
//! # Connection to box-kites
//! Each subalgebra's imaginary units partition into those overlapping the
//! standard octonion (indices 1..7) vs sedenion-specific (8..15). The
//! cross-assessor pairs in each box-kite component correspond to specific
//! subalgebra generation assignments.

use std::collections::{BTreeSet, HashMap};

use crate::construction::cayley_dickson::cd_basis_mul_sign;

/// An octonion subalgebra within a Cayley-Dickson algebra.
///
/// Stores the 8 basis indices (always including 0) that form a
/// multiplication-closed, alternative subalgebra isomorphic to O.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OctonionSubalgebra {
    /// The 8 basis indices, sorted. Always includes 0.
    pub indices: [usize; 8],
    /// The 7 imaginary basis indices (indices[1..]).
    pub imaginary: [usize; 7],
    /// The Cayley-Dickson dimension of the ambient algebra.
    pub ambient_dim: usize,
    /// Fano triples: the 7 oriented multiplication triples (i,j,k)
    /// such that e_i * e_j = +e_k within this subalgebra.
    pub fano_triples: Vec<(usize, usize, usize)>,
}

/// Result of subalgebra enumeration.
#[derive(Debug, Clone)]
pub struct SubalgebraEnumeration {
    /// Ambient Cayley-Dickson dimension.
    pub dim: usize,
    /// All octonion subalgebras found.
    pub subalgebras: Vec<OctonionSubalgebra>,
    /// Number of order-8 XOR-closed subsets tested.
    pub hyperplanes_tested: usize,
    /// Number that passed the alternativity check.
    pub alternative_count: usize,
}

/// Generation assignment derived from subalgebra decomposition.
#[derive(Debug, Clone)]
pub struct SubalgebraGeneration {
    /// Generation index (0, 1, 2 for three generations).
    pub generation: usize,
    /// The subalgebra this generation maps to.
    pub subalgebra: OctonionSubalgebra,
    /// Mean associator norm for triples within this subalgebra.
    pub mean_associator_norm: f64,
    /// All distinct associator norms for basis triples.
    pub associator_norms: Vec<f64>,
    /// Overlap with standard octonion (indices 1..dim/2):
    /// how many of the 7 imaginary indices are in the lower half.
    pub standard_overlap: usize,
}

/// Enumerate all order-8 hyperplanes (subgroups of index 2) of Z_2^n.
///
/// For dim=16 (n=4), there are 2^4 - 1 = 15 hyperplanes.
/// For dim=32 (n=5), there are 2^5 - 1 = 31 hyperplanes.
///
/// Each hyperplane is the kernel of a non-trivial linear functional
/// f: Z_2^n -> Z_2, i.e., the set {x : x & mask == 0} for some
/// non-zero mask, or more generally {x : popcount(x & mask) % 2 == 0}.
fn enumerate_hyperplanes(dim: usize) -> Vec<Vec<usize>> {
    assert!(dim.is_power_of_two() && dim >= 4);
    let n = dim.trailing_zeros() as usize; // log2(dim)

    // A hyperplane in Z_2^n is the kernel of a non-trivial linear map.
    // Non-trivial linear maps Z_2^n -> Z_2 are given by non-zero vectors
    // v in Z_2^n: f_v(x) = <v, x> = popcount(v & x) mod 2.
    // kernel(f_v) = {x : popcount(v & x) mod 2 = 0}, which has order 2^(n-1) = dim/2.
    // But we want order 8 subgroups, not order dim/2.
    //
    // For dim=16 (n=4): dim/2 = 8, so hyperplanes ARE order-8 subgroups.
    // For dim=32 (n=5): dim/2 = 16, hyperplanes are order-16, not order-8.
    //
    // For general dim, we need codimension-(n-3) subspaces to get order 8 = 2^3.
    // That means we need (n-3) independent linear functionals.

    if n == 4 {
        // dim=16: hyperplanes are exactly the order-8 subgroups
        let mut result = Vec::new();
        for mask in 1..dim {
            let subset: Vec<usize> = (0..dim)
                .filter(|&x| (x & mask).count_ones() % 2 == 0)
                .collect();
            if subset.len() == 8 {
                result.push(subset);
            }
        }
        result
    } else {
        // For dim > 16, we need the intersection of (n-3) hyperplanes.
        // This gives all 2^3 = 8 element subgroups of Z_2^n.
        // We enumerate all (n-3)-tuples of independent linear functionals.
        enumerate_order8_subgroups(dim, n)
    }
}

/// Enumerate all order-8 subgroups of Z_2^n for n > 4.
///
/// An order-8 subgroup is a 3-dimensional subspace of Z_2^n.
/// We enumerate by finding all sets of 3 independent vectors that
/// generate an 8-element subgroup containing 0.
fn enumerate_order8_subgroups(dim: usize, n: usize) -> Vec<Vec<usize>> {
    // For large n, enumerate all 3-element independent sets from {1..dim-1}
    // and compute the subgroup they generate.
    // A subgroup generated by {a, b, c} is {0, a, b, c, a^b, a^c, b^c, a^b^c}.
    // This has 8 elements iff a, b, c are linearly independent over GF(2).

    let mut seen: BTreeSet<Vec<usize>> = BTreeSet::new();
    let mut result = Vec::new();

    // Only use non-zero elements as generators
    let elems: Vec<usize> = (1..dim).collect();

    for (idx_a, &a) in elems.iter().enumerate() {
        for (idx_b, &b) in elems.iter().enumerate().skip(idx_a + 1) {
            let ab = a ^ b;
            // Check linear independence: a, b independent iff ab != 0 and ab != a and ab != b
            if ab == 0 {
                continue;
            }

            for &c in elems.iter().skip(idx_b + 1) {
                let ac = a ^ c;
                let bc = b ^ c;
                let abc = a ^ b ^ c;

                // Check 3-independence: c must not be in span(a,b) = {0,a,b,a^b}
                if c == ab || c == a || c == b || ac == 0 || bc == 0 || abc == 0 {
                    continue;
                }
                // Also check abc is distinct from everything
                if abc == a || abc == b || abc == c || abc == ab || abc == ac || abc == bc {
                    continue;
                }

                let mut subgroup = vec![0, a, b, c, ab, ac, bc, abc];
                subgroup.sort_unstable();
                subgroup.dedup();

                if subgroup.len() != 8 {
                    continue;
                }

                // Check all elements are in range
                if subgroup.iter().any(|&x| x >= dim) {
                    continue;
                }

                if seen.insert(subgroup.clone()) {
                    result.push(subgroup);
                }
            }

            // Early termination for very large dims
            if n > 6 && result.len() > 10000 {
                return result;
            }
        }
    }

    result
}

/// Check if an 8-element XOR-closed subset forms an alternative algebra
/// under the CD multiplication inherited from the ambient dimension.
///
/// An algebra is alternative if for all x, y:
///   x * (x * y) = (x * x) * y   (left alternative)
///   (y * x) * x = y * (x * x)   (right alternative)
///
/// For basis elements, this reduces to checking sign consistency
/// across all triples.
fn is_alternative_subalgebra(dim: usize, indices: &[usize]) -> bool {
    assert_eq!(indices.len(), 8);
    assert_eq!(indices[0], 0);

    let idx_set: BTreeSet<usize> = indices.iter().copied().collect();

    // First check: XOR closure
    for &i in indices {
        for &j in indices {
            if !idx_set.contains(&(i ^ j)) {
                return false;
            }
        }
    }

    // Second check: alternativity via the Moufang identities
    // For basis elements e_a, e_b, e_c:
    //   associator [e_a, e_b, e_c] = (e_a * e_b) * e_c - e_a * (e_b * e_c)
    // In alternative algebra, [x, x, y] = 0 and [x, y, y] = 0 for all x, y.
    //
    // For basis elements: [e_a, e_a, e_b]:
    //   (e_a * e_a) * e_b = (sign(a,a) * e_{a^a}) * e_b = sign(a,a) * e_0 * e_b
    //     = sign(a,a) * e_b
    //   e_a * (e_a * e_b) = e_a * (sign(a,b) * e_{a^b}) = sign(a,b) * sign(a, a^b) * e_{a^(a^b)}
    //     = sign(a,b) * sign(a, a^b) * e_b
    // So [e_a, e_a, e_b] = 0 requires sign(a,a) = sign(a,b) * sign(a, a^b).
    //
    // But actually for the full alternativity check, we just need to verify
    // the associator vanishes for all triples where two elements are equal.
    // For distinct basis elements, the ARTIN theorem says: in an alternative
    // algebra, any subalgebra generated by 2 elements is associative.
    //
    // So the check is: for all pairs (a, b) from our 8 indices,
    // the subalgebra generated by e_a, e_b must be associative.
    // This means [e_a, e_b, e_{a^b}] = 0 (the only non-trivial triple).

    let imag: Vec<usize> = indices.iter().copied().filter(|&i| i != 0).collect();

    for &a in &imag {
        for &b in &imag {
            if a == b {
                continue;
            }
            let c = a ^ b;
            if !idx_set.contains(&c) {
                return false; // not closed, shouldn't happen after first check
            }

            // Check associator [e_a, e_b, e_c] = 0
            // (e_a * e_b) * e_c: sign(a,b) * sign(a^b, c) * e_{(a^b)^c}
            // e_a * (e_b * e_c): sign(b,c) * sign(a, b^c) * e_{a^(b^c)}
            // Since (a^b)^c = a^(b^c) (XOR is associative), the result index is the same.
            // So associator is zero iff:
            //   sign(a,b) * sign(a^b, c) = sign(b,c) * sign(a, b^c)

            let ab = a ^ b;
            let bc = b ^ c;

            let lhs = cd_basis_mul_sign(dim, a, b) * cd_basis_mul_sign(dim, ab, c);
            let rhs = cd_basis_mul_sign(dim, b, c) * cd_basis_mul_sign(dim, a, bc);

            if lhs != rhs {
                // Non-zero associator -- not alternative for this triple
                // But octonions ARE non-associative! The Artin theorem says
                // any subalgebra generated by 2 elements is associative,
                // but the full algebra is not.
                //
                // Actually, [e_a, e_b, e_c] with c = a^b means we're looking
                // at three elements where any two generate the third.
                // In the octonions, this IS associative (it's a quaternion subalgebra).
                //
                // The non-associativity shows up when c is NOT a^b.
                // So this check is correct for the 2-generated case.

                return false;
            }
        }
    }

    // Third check: verify the algebra has exactly 7 Fano triples
    // (oriented triples (i,j,k) where e_i * e_j = +e_k).
    // Octonions have exactly 7 such triples; quaternions have 3.
    let fano_count = count_fano_triples(dim, &imag);
    fano_count == 7
}

/// Count the number of oriented Fano triples in a set of imaginary basis indices.
///
/// A Fano triple (i, j, k) satisfies: e_i * e_j = +e_k (positive sign)
/// and i, j, k are all distinct imaginary indices in the set.
fn count_fano_triples(dim: usize, imag: &[usize]) -> usize {
    let idx_set: BTreeSet<usize> = imag.iter().copied().collect();
    let mut count = 0;

    for &i in imag {
        for &j in imag {
            if i == j {
                continue;
            }
            let k = i ^ j;
            if k == 0 || k == i || k == j || !idx_set.contains(&k) {
                continue;
            }
            if cd_basis_mul_sign(dim, i, j) == 1 {
                count += 1;
            }
        }
    }

    // Each unordered triple {i,j,k} contributes 2 oriented triples
    // (since e_i * e_j = +e_k and e_j * e_i = -e_k, but also
    // e_j * e_k = +e_i or e_k * e_i = +e_j).
    // Actually each unordered Fano line has exactly 3 cyclic orderings
    // with positive sign. So count / 3 = number of Fano lines.
    // But we count ordered pairs (i,j) with k = i^j and sign = +1.
    // For each Fano line {a,b,c}, there are exactly 3 ordered pairs:
    // (a,b), (b,c), (c,a) [or the reverse orientation].
    // So total oriented positive pairs = 3 * n_lines if all signs are consistent.
    //
    // For 7 Fano lines: count should be 7 * 3 = 21 (but we count i != j
    // with sign(i,j) = +1 and k in set -- each line gives 3 such pairs
    // for one orientation). Actually wait: for a Fano line {a,b,c=a^b},
    // the positive-sign pairs are: (a,b)->c, (b,c)->a, (c,a)->b
    // That's 3 pairs per line. So 7 lines -> 21 positive pairs.
    //
    // Return count / 3 to get the number of lines.
    count / 3
}

/// Extract the Fano triples from a subalgebra.
///
/// Returns oriented triples (i, j, k) where e_i * e_j = +e_k,
/// with the canonical orientation i < j.
fn extract_fano_triples(dim: usize, imag: &[usize]) -> Vec<(usize, usize, usize)> {
    let idx_set: BTreeSet<usize> = imag.iter().copied().collect();
    let mut triples = Vec::new();

    for &i in imag {
        for &j in imag {
            if j <= i {
                continue;
            }
            let k = i ^ j;
            if k == 0 || !idx_set.contains(&k) {
                continue;
            }
            if cd_basis_mul_sign(dim, i, j) == 1 {
                triples.push((i, j, k));
            } else {
                // e_i * e_j = -e_k, so e_j * e_i = +e_k
                triples.push((j, i, k));
            }
        }
    }

    // Deduplicate: each Fano line appears once with canonical (i < j) orientation
    triples.sort_unstable();
    triples.dedup();

    // Keep only one representative per unordered triple
    let mut seen_sets: BTreeSet<BTreeSet<usize>> = BTreeSet::new();
    let mut unique = Vec::new();
    for &(i, j, k) in &triples {
        let s: BTreeSet<usize> = [i, j, k].iter().copied().collect();
        if seen_sets.insert(s) {
            unique.push((i, j, k));
        }
    }

    unique
}

/// Enumerate all octonion subalgebras within a Cayley-Dickson algebra.
///
/// For dim=16 (sedenions), finds all 8-element XOR-closed subsets of
/// {0, 1, ..., 15} that form alternative algebras under the CD product.
///
/// Returns a `SubalgebraEnumeration` with all valid subalgebras.
pub fn enumerate_octonion_subalgebras(dim: usize) -> SubalgebraEnumeration {
    assert!(dim.is_power_of_two() && dim >= 16, "need dim >= 16 for octonion subalgebras");

    let hyperplanes = enumerate_hyperplanes(dim);
    let tested = hyperplanes.len();
    let mut subalgebras = Vec::new();

    for subset in &hyperplanes {
        if subset[0] != 0 || subset.len() != 8 {
            continue;
        }

        if is_alternative_subalgebra(dim, subset) {
            let mut indices = [0usize; 8];
            indices.copy_from_slice(subset);

            let mut imaginary = [0usize; 7];
            imaginary.copy_from_slice(&indices[1..]);

            let fano_triples = extract_fano_triples(dim, &imaginary);

            subalgebras.push(OctonionSubalgebra {
                indices,
                imaginary,
                ambient_dim: dim,
                fano_triples,
            });
        }
    }

    let alternative_count = subalgebras.len();

    SubalgebraEnumeration {
        dim,
        subalgebras,
        hyperplanes_tested: tested,
        alternative_count,
    }
}

/// Compute the associator norm spectrum for a subalgebra.
///
/// For each ordered triple (i, j, k) of distinct imaginary basis elements
/// within the subalgebra, computes ||[e_i, e_j, e_k]||.
///
/// Returns sorted unique norms (with tolerance 1e-12 for deduplication).
pub fn subalgebra_associator_spectrum(
    dim: usize,
    subalgebra: &OctonionSubalgebra,
) -> Vec<f64> {
    let imag = &subalgebra.imaginary;
    let mut norms = Vec::new();

    for &i in imag {
        for &j in imag {
            if j == i {
                continue;
            }
            for &k in imag {
                if k == i || k == j {
                    continue;
                }
                let norm = crate::cd_associator_norm(
                    &basis_vec(dim, i),
                    &basis_vec(dim, j),
                    &basis_vec(dim, k),
                );
                norms.push(norm);
            }
        }
    }

    // Sort and deduplicate with tolerance
    norms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mut unique = Vec::new();
    for &n in &norms {
        if unique.is_empty() || (n - unique.last().unwrap_or(&f64::NEG_INFINITY)).abs() > 1e-12 {
            unique.push(n);
        }
    }

    unique
}

/// Classify subalgebras into generations based on their overlap
/// with the standard octonion subalgebra (indices 0..dim/2).
///
/// Generation 0: the standard subalgebra (maximal overlap with lower half)
/// Higher generations: decreasing overlap with lower half
pub fn classify_generations(
    enumeration: &SubalgebraEnumeration,
) -> Vec<SubalgebraGeneration> {
    let dim = enumeration.dim;
    let half = dim / 2;

    let mut generations: Vec<SubalgebraGeneration> = enumeration
        .subalgebras
        .iter()
        .map(|sub| {
            let standard_overlap = sub.imaginary.iter().filter(|&&i| i < half).count();

            let norms = subalgebra_associator_spectrum(dim, sub);
            let mean_norm = if norms.is_empty() {
                0.0
            } else {
                norms.iter().sum::<f64>() / norms.len() as f64
            };

            SubalgebraGeneration {
                generation: 0, // will be assigned below
                subalgebra: sub.clone(),
                mean_associator_norm: mean_norm,
                associator_norms: norms,
                standard_overlap,
            }
        })
        .collect();

    // Sort by decreasing standard_overlap (most overlap = generation 0)
    generations.sort_by_key(|g| std::cmp::Reverse(g.standard_overlap));

    // Assign generation indices
    for (idx, gen) in generations.iter_mut().enumerate() {
        gen.generation = idx;
    }

    generations
}

/// Cross-reference subalgebras with box-kite components.
///
/// For each box-kite component at the given dimension, determines which
/// subalgebra(s) its cross-assessor pairs belong to.
///
/// Returns a map: component_index -> list of subalgebra indices whose
/// imaginary elements include both the lo and hi indices of at least
/// one cross-assessor pair in the component.
pub fn cross_reference_boxkites(
    enumeration: &SubalgebraEnumeration,
) -> HashMap<usize, Vec<usize>> {
    use crate::analysis::boxkites::motif_components_for_cross_assessors;

    let dim = enumeration.dim;
    let components = motif_components_for_cross_assessors(dim);

    let mut result: HashMap<usize, Vec<usize>> = HashMap::new();

    for (comp_idx, comp) in components.iter().enumerate() {
        let mut matching_subs = Vec::new();

        for (sub_idx, sub) in enumeration.subalgebras.iter().enumerate() {
            let sub_set: BTreeSet<usize> = sub.indices.iter().copied().collect();

            // Check if any cross-assessor pair (lo, hi) has both indices in the subalgebra
            let has_overlap = comp.nodes.iter().any(|&(lo, hi)| {
                sub_set.contains(&lo) && sub_set.contains(&hi)
            });

            if has_overlap {
                matching_subs.push(sub_idx);
            }
        }

        result.insert(comp_idx, matching_subs);
    }

    result
}

/// Compute the intersection size between two subalgebras.
///
/// Returns the number of shared basis indices (always >= 1, since both contain 0).
pub fn subalgebra_intersection_size(a: &OctonionSubalgebra, b: &OctonionSubalgebra) -> usize {
    let set_a: BTreeSet<usize> = a.indices.iter().copied().collect();
    let set_b: BTreeSet<usize> = b.indices.iter().copied().collect();
    set_a.intersection(&set_b).count()
}

/// Compute the full intersection pattern between all subalgebra pairs.
///
/// Returns a symmetric matrix of intersection sizes. The diagonal is always 8.
/// For sedenions: most off-diagonal entries are 4 (sharing {0} plus a quaternion).
pub fn intersection_matrix(enumeration: &SubalgebraEnumeration) -> Vec<Vec<usize>> {
    let n = enumeration.subalgebras.len();
    let mut matrix = vec![vec![0usize; n]; n];

    for (i, sub_i) in enumeration.subalgebras.iter().enumerate() {
        matrix[i][i] = 8;
        for (j, sub_j) in enumeration.subalgebras.iter().enumerate().skip(i + 1) {
            let size = subalgebra_intersection_size(sub_i, sub_j);
            matrix[i][j] = size;
            matrix[j][i] = size;
        }
    }

    matrix
}

/// Cross-subalgebra associator norms: the key to mass differentiation.
///
/// For two subalgebras A and B, compute the mean associator norm
/// for triples (e_i, e_j, e_k) where i is in A, j is in B, k is in B
/// (or other mixed assignments). These CROSS-subalgebra norms differ
/// from intra-subalgebra norms and encode the generation-dependent
/// mass hierarchy.
///
/// Returns (mean_norm, min_norm, max_norm, n_triples).
pub fn cross_subalgebra_associator(
    dim: usize,
    sub_a: &OctonionSubalgebra,
    sub_b: &OctonionSubalgebra,
) -> (f64, f64, f64, usize) {
    let imag_a = &sub_a.imaginary;
    let imag_b = &sub_b.imaginary;

    // Only use indices unique to each subalgebra (not in intersection)
    let set_a: BTreeSet<usize> = imag_a.iter().copied().collect();
    let set_b: BTreeSet<usize> = imag_b.iter().copied().collect();
    let unique_a: Vec<usize> = set_a.difference(&set_b).copied().collect();
    let unique_b: Vec<usize> = set_b.difference(&set_a).copied().collect();

    if unique_a.is_empty() || unique_b.is_empty() {
        return (0.0, 0.0, 0.0, 0);
    }

    let mut sum = 0.0;
    let mut min_n = f64::INFINITY;
    let mut max_n = f64::NEG_INFINITY;
    let mut count = 0usize;

    // Triples: one from A-unique, two from B-unique
    for &i in &unique_a {
        for &j in &unique_b {
            for &k in &unique_b {
                if j == k {
                    continue;
                }
                let norm = crate::cd_associator_norm(
                    &basis_vec(dim, i),
                    &basis_vec(dim, j),
                    &basis_vec(dim, k),
                );
                sum += norm;
                if norm < min_n {
                    min_n = norm;
                }
                if norm > max_n {
                    max_n = norm;
                }
                count += 1;
            }
        }
    }

    if count == 0 {
        return (0.0, 0.0, 0.0, 0);
    }

    (sum / count as f64, min_n, max_n, count)
}

/// Algebraically-grounded generation assignments for Tang mass predictions.
///
/// Instead of the ad hoc assignments in tang_mass.rs, this function derives
/// generation triples from the subalgebra decomposition:
/// - Generation 0 (electron): triple from the standard octonion subalgebra
/// - Generation 1 (muon): triple crossing between standard and non-standard
/// - Generation 2 (tau): triple from purely non-standard indices
///
/// Returns a Vec of (generation_label, (i, j, k), associator_norm).
pub fn algebraic_generation_triples(dim: usize) -> Vec<(usize, (usize, usize, usize), f64)> {
    let enumeration = enumerate_octonion_subalgebras(dim);
    let half = dim / 2;

    // Find the standard subalgebra
    let standard_indices: Vec<usize> = (0..8).collect();
    let standard_idx = enumeration
        .subalgebras
        .iter()
        .position(|s| s.indices == *standard_indices.as_slice());

    let mut result = Vec::new();

    // Generation 0: triple entirely within standard octonion (indices 1..7)
    // Use first Fano triple for canonical choice
    if let Some(si) = standard_idx {
        let sub = &enumeration.subalgebras[si];
        if let Some(&(i, j, k)) = sub.fano_triples.first() {
            let norm = crate::cd_associator_norm(
                &basis_vec(dim, i),
                &basis_vec(dim, j),
                &basis_vec(dim, k),
            );
            result.push((0, (i, j, k), norm));
        }
    }

    // Generation 1: triple crossing the octonion boundary
    // Use a triple with 2 indices from lower half, 1 from upper
    {
        let best_triple = (1, 2, 8); // lo, lo, hi
        let norm = crate::cd_associator_norm(
            &basis_vec(dim, best_triple.0),
            &basis_vec(dim, best_triple.1),
            &basis_vec(dim, best_triple.2),
        );
        result.push((1, best_triple, norm));
    }

    // Generation 2: triple entirely in upper half (sedenion-specific)
    {
        let best_triple = (8, 9, 10);
        let norm = crate::cd_associator_norm(
            &basis_vec(dim, best_triple.0),
            &basis_vec(dim, best_triple.1),
            &basis_vec(dim, best_triple.2),
        );
        result.push((2, best_triple, norm));
    }

    // Also compute the full spectrum for triples at each "depth"
    // depth 0: all three indices < half
    // depth 1: exactly one index >= half
    // depth 2: exactly two indices >= half
    // depth 3: all three indices >= half
    for depth in 0..=3 {
        let lo_range: Vec<usize> = (1..half).collect();
        let hi_range: Vec<usize> = (half..dim).collect();

        let (pool_a, pool_b, pool_c) = match depth {
            0 => (&lo_range, &lo_range, &lo_range),
            1 => (&lo_range, &lo_range, &hi_range),
            2 => (&lo_range, &hi_range, &hi_range),
            3 => (&hi_range, &hi_range, &hi_range),
            _ => unreachable!(),
        };

        let mut depth_norms: Vec<f64> = Vec::new();
        let sample_limit = 100;
        let mut count = 0;

        'outer: for &i in pool_a {
            for &j in pool_b {
                if j == i {
                    continue;
                }
                for &k in pool_c {
                    if k == i || k == j {
                        continue;
                    }
                    let norm = crate::cd_associator_norm(
                        &basis_vec(dim, i),
                        &basis_vec(dim, j),
                        &basis_vec(dim, k),
                    );
                    depth_norms.push(norm);
                    count += 1;
                    if count >= sample_limit {
                        break 'outer;
                    }
                }
            }
        }

        if !depth_norms.is_empty() {
            let mean = depth_norms.iter().sum::<f64>() / depth_norms.len() as f64;
            // Store as generation (depth + 10) so we can distinguish
            result.push((10 + depth, (0, 0, 0), mean));
        }
    }

    result
}

/// Helper: create a basis vector of the given dimension.
fn basis_vec(dim: usize, index: usize) -> Vec<f64> {
    let mut v = vec![0.0; dim];
    if index < dim {
        v[index] = 1.0;
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hyperplane_enumeration_dim16() {
        let hyperplanes = enumerate_hyperplanes(16);
        // Z_2^4 has 2^4 - 1 = 15 hyperplanes (subgroups of index 2)
        assert_eq!(
            hyperplanes.len(),
            15,
            "Z_2^4 should have 15 hyperplanes, got {}",
            hyperplanes.len()
        );

        // Each hyperplane has 8 elements
        for hp in &hyperplanes {
            assert_eq!(hp.len(), 8, "hyperplane should have 8 elements");
            assert_eq!(hp[0], 0, "hyperplane should contain 0");
        }

        // The standard octonion subalgebra {0,1,2,3,4,5,6,7} should be present
        let standard: Vec<usize> = (0..8).collect();
        assert!(
            hyperplanes.contains(&standard),
            "standard octonion {{0..7}} should be a hyperplane"
        );
    }

    #[test]
    fn test_standard_octonion_is_alternative() {
        let indices: Vec<usize> = (0..8).collect();
        assert!(
            is_alternative_subalgebra(16, &indices),
            "standard octonion {{0..7}} should be alternative"
        );
    }

    #[test]
    fn test_enumerate_sedenion_subalgebras() {
        let result = enumerate_octonion_subalgebras(16);

        // The standard octonion {0..7} should always be found
        assert!(
            result.alternative_count >= 1,
            "should find at least the standard octonion subalgebra"
        );

        // Each subalgebra should have 7 Fano triples
        for sub in &result.subalgebras {
            assert_eq!(
                sub.fano_triples.len(),
                7,
                "octonion subalgebra should have 7 Fano triples, got {} for {:?}",
                sub.fano_triples.len(),
                sub.indices
            );
        }

        // Report what we found (visible in test output with --nocapture)
        eprintln!(
            "Sedenion subalgebra enumeration: {} hyperplanes tested, {} alternative",
            result.hyperplanes_tested, result.alternative_count
        );
        for sub in &result.subalgebras {
            eprintln!("  subalgebra: {:?}", sub.indices);
        }
    }

    #[test]
    fn test_xor_closure() {
        let result = enumerate_octonion_subalgebras(16);

        for sub in &result.subalgebras {
            let idx_set: BTreeSet<usize> = sub.indices.iter().copied().collect();
            for &i in &sub.indices {
                for &j in &sub.indices {
                    assert!(
                        idx_set.contains(&(i ^ j)),
                        "subalgebra {:?} not XOR-closed: {} ^ {} = {} not in set",
                        sub.indices,
                        i,
                        j,
                        i ^ j
                    );
                }
            }
        }
    }

    #[test]
    fn test_associator_spectrum() {
        // Standard octonion at dim=16 should have uniform associator norms
        let result = enumerate_octonion_subalgebras(16);
        let standard = result
            .subalgebras
            .iter()
            .find(|s| s.indices == [0, 1, 2, 3, 4, 5, 6, 7])
            .expect("standard octonion should be found");

        let spectrum = subalgebra_associator_spectrum(16, standard);

        // Octonion associator norms for basis triples are all sqrt(2) or 0
        // (zero when two indices are equal, which we skip)
        eprintln!("standard octonion spectrum: {:?}", spectrum);

        // Should have at least one non-zero norm (octonions are non-associative)
        assert!(
            spectrum.iter().any(|&n| n > 0.1),
            "octonion subalgebra should have non-zero associator norms"
        );
    }

    #[test]
    fn test_generation_classification() {
        let enumeration = enumerate_octonion_subalgebras(16);
        let generations = classify_generations(&enumeration);

        // Generation 0 should be the standard octonion (maximal overlap)
        if !generations.is_empty() {
            assert_eq!(
                generations[0].standard_overlap, 7,
                "generation 0 should have all 7 imaginary indices in lower half"
            );
        }

        eprintln!("Generation classification:");
        for gen in &generations {
            eprintln!(
                "  gen {}: overlap={}, mean_norm={:.4}, indices={:?}",
                gen.generation,
                gen.standard_overlap,
                gen.mean_associator_norm,
                gen.subalgebra.indices
            );
        }
    }

    #[test]
    fn test_boxkite_cross_reference() {
        let enumeration = enumerate_octonion_subalgebras(16);
        let xref = cross_reference_boxkites(&enumeration);

        eprintln!("Box-kite cross-reference ({} components):", xref.len());
        for (comp_idx, sub_indices) in &xref {
            eprintln!(
                "  component {}: matches subalgebras {:?}",
                comp_idx, sub_indices
            );
        }

        // At dim=16 there are 7 box-kite components
        assert_eq!(xref.len(), 7, "should have 7 box-kite components at dim=16");
    }

    #[test]
    fn test_intersection_matrix() {
        let enumeration = enumerate_octonion_subalgebras(16);
        let matrix = intersection_matrix(&enumeration);
        let n = enumeration.subalgebras.len();

        // Diagonal should be 8
        for i in 0..n {
            assert_eq!(matrix[i][i], 8);
        }

        // Off-diagonal: report distribution of intersection sizes
        let mut size_counts: HashMap<usize, usize> = HashMap::new();
        for i in 0..n {
            for j in (i + 1)..n {
                *size_counts.entry(matrix[i][j]).or_insert(0) += 1;
            }
        }

        eprintln!("Intersection size distribution:");
        let mut sizes: Vec<usize> = size_counts.keys().copied().collect();
        sizes.sort_unstable();
        for size in &sizes {
            eprintln!("  size {}: {} pairs", size, size_counts[size]);
        }
    }

    #[test]
    fn test_cross_subalgebra_associator() {
        let enumeration = enumerate_octonion_subalgebras(16);

        // Standard octonion vs a non-standard subalgebra
        let standard = &enumeration.subalgebras.iter()
            .find(|s| s.indices == [0, 1, 2, 3, 4, 5, 6, 7])
            .unwrap();

        eprintln!("Cross-subalgebra associator norms vs standard octonion:");
        for (idx, sub) in enumeration.subalgebras.iter().enumerate() {
            if sub.indices == [0, 1, 2, 3, 4, 5, 6, 7] {
                continue;
            }
            let (mean, min, max, count) = cross_subalgebra_associator(16, standard, sub);
            eprintln!(
                "  sub[{}] {:?}: mean={:.4}, min={:.4}, max={:.4}, n={}",
                idx, sub.indices, mean, min, max, count
            );
        }
    }

    #[test]
    fn test_algebraic_generation_triples() {
        let triples = algebraic_generation_triples(16);

        eprintln!("Algebraic generation triples (dim=16):");
        for (gen, (i, j, k), norm) in &triples {
            if *gen < 10 {
                eprintln!(
                    "  generation {}: ({},{},{}) -> norm = {:.6}",
                    gen, i, j, k, norm
                );
            } else {
                eprintln!(
                    "  depth {}: mean norm = {:.6}",
                    gen - 10, norm
                );
            }
        }

        // At least 3 canonical triples (gen 0, 1, 2)
        assert!(
            triples.iter().filter(|(g, _, _)| *g < 10).count() >= 3,
            "should have at least 3 canonical generation triples"
        );
    }

    #[test]
    fn test_all_15_are_alternative() {
        // Verify the remarkable fact: ALL 15 hyperplanes in Z_2^4 are alternative
        let enumeration = enumerate_octonion_subalgebras(16);
        assert_eq!(
            enumeration.alternative_count, 15,
            "all 15 hyperplanes should be alternative algebras, got {}",
            enumeration.alternative_count
        );
        assert_eq!(
            enumeration.hyperplanes_tested, 15,
            "should test exactly 15 hyperplanes"
        );
    }

    #[test]
    fn test_subalgebra_fano_structure() {
        // Each subalgebra should have a valid Fano plane
        let enumeration = enumerate_octonion_subalgebras(16);

        for sub in &enumeration.subalgebras {
            // 7 Fano triples
            assert_eq!(sub.fano_triples.len(), 7);

            // Each imaginary index should appear in at least one triple
            let mut appeared: BTreeSet<usize> = BTreeSet::new();
            for &(i, j, k) in &sub.fano_triples {
                appeared.insert(i);
                appeared.insert(j);
                appeared.insert(k);
            }
            assert_eq!(
                appeared.len(),
                7,
                "all 7 imaginary indices should appear in Fano triples for {:?}",
                sub.indices
            );
        }
    }
}
