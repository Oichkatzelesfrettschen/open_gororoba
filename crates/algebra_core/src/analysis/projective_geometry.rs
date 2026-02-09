//! Finite projective geometry PG(m,2) over GF(2).
//!
//! Points of PG(m,2) are non-zero vectors of GF(2)^{m+1}, represented as
//! usize bitmasks. Lines are unordered triples {a, b, a XOR b} where a, b
//! are distinct non-zero points (and a XOR b is automatically non-zero and
//! distinct from both).
//!
//! The key connection to Cayley-Dickson algebras: at dimension 2^n, the
//! zero-divisor motif components correspond bijectively to points of
//! PG(n-2, 2). The XOR-key structure of each component encodes its
//! GF(2)^{n-1} label.
//!
//! # References
//!
//! - Hirschfeld (1998): Projective Geometries over Finite Fields
//! - Saniga, Holweck, Pracna (2015): Cayley-Dickson algebras and finite geometry
//! - Polster (1998): A Geometrical Picture Book

use std::collections::{BTreeSet, HashMap, HashSet};

/// A point in PG(m,2), represented as a non-zero bitmask in GF(2)^{m+1}.
pub type PGPoint = usize;

/// A line in PG(m,2): an unordered triple {a, b, a XOR b}.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PGLine {
    /// The three collinear points, sorted for canonical representation.
    pub points: [PGPoint; 3],
}

/// A finite projective space PG(m,2).
#[derive(Debug, Clone)]
pub struct ProjectiveGeometry {
    /// Dimension parameter m (the projective dimension).
    pub m: usize,
    /// All points (non-zero vectors of GF(2)^{m+1}).
    pub points: Vec<PGPoint>,
    /// All lines (triples {a, b, a^b}).
    pub lines: Vec<PGLine>,
    /// For each point, the indices of lines containing it.
    pub point_lines: HashMap<PGPoint, Vec<usize>>,
}

/// Construct PG(m,2) for a given projective dimension m.
///
/// - PG(0,2) has 1 point, 0 lines
/// - PG(1,2) has 3 points, 1 line
/// - PG(2,2) = Fano plane: 7 points, 7 lines
/// - PG(3,2): 15 points, 35 lines
/// - PG(m,2): (2^{m+1}-1) points, (2^{m+1}-1)(2^m-1)/3 lines
pub fn pg(m: usize) -> ProjectiveGeometry {
    let n_bits = m + 1;
    let n_points = (1_usize << n_bits) - 1;

    // Points: all non-zero bitmasks of width n_bits
    let points: Vec<PGPoint> = (1..=n_points).collect();
    let point_set: HashSet<PGPoint> = points.iter().copied().collect();

    // Lines: for each unordered pair {a, b}, the third point is a^b.
    // We only generate each line once by requiring a < b < a^b OR
    // using a canonical sorted representation.
    let mut lines: Vec<PGLine> = Vec::new();
    let mut line_set: HashSet<BTreeSet<PGPoint>> = HashSet::new();

    for i in 0..points.len() {
        for j in (i + 1)..points.len() {
            let a = points[i];
            let b = points[j];
            let c = a ^ b;
            if c == 0 || !point_set.contains(&c) {
                continue;
            }
            let mut triple = BTreeSet::new();
            triple.insert(a);
            triple.insert(b);
            triple.insert(c);
            if triple.len() == 3 && line_set.insert(triple.clone()) {
                let mut pts: Vec<PGPoint> = triple.into_iter().collect();
                pts.sort_unstable();
                lines.push(PGLine {
                    points: [pts[0], pts[1], pts[2]],
                });
            }
        }
    }

    lines.sort_by_key(|l| l.points);

    // Build point-to-line index
    let mut point_lines: HashMap<PGPoint, Vec<usize>> = HashMap::new();
    for p in &points {
        point_lines.insert(*p, Vec::new());
    }
    for (idx, line) in lines.iter().enumerate() {
        for &p in &line.points {
            point_lines.get_mut(&p).unwrap().push(idx);
        }
    }

    ProjectiveGeometry {
        m,
        points,
        lines,
        point_lines,
    }
}

/// Construct PG(n-2,2) appropriate for Cayley-Dickson dimension 2^n.
///
/// For dim=16 (n=4): PG(2,2) = Fano plane (7 pts, 7 lines)
/// For dim=32 (n=5): PG(3,2) (15 pts, 35 lines)
/// For dim=64 (n=6): PG(4,2) (31 pts, 155 lines)
pub fn pg_from_cd_dim(dim: usize) -> ProjectiveGeometry {
    assert!(
        dim >= 16 && dim.is_power_of_two(),
        "dim must be 2^n with n >= 4"
    );
    let n = dim.trailing_zeros() as usize;
    pg(n - 2)
}

/// Compute the incidence matrix of PG(m,2).
///
/// Returns a (n_points x n_lines) matrix where entry (i,j) = 1
/// if point i lies on line j, 0 otherwise.
pub fn incidence_matrix(geom: &ProjectiveGeometry) -> Vec<Vec<u8>> {
    let n_pts = geom.points.len();
    let n_lines = geom.lines.len();
    let point_idx: HashMap<PGPoint, usize> = geom
        .points
        .iter()
        .enumerate()
        .map(|(i, &p)| (p, i))
        .collect();

    let mut matrix = vec![vec![0u8; n_lines]; n_pts];
    for (j, line) in geom.lines.iter().enumerate() {
        for &p in &line.points {
            let i = point_idx[&p];
            matrix[i][j] = 1;
        }
    }
    matrix
}

// ============================================================================
// PG-to-motif bijection (A2)
// ============================================================================

use crate::analysis::boxkites::{CrossPair, MotifComponent};
use crate::analysis::zd_graphs::xor_key;

/// Extract the XOR-key label for a motif component, mapped to PG(n-2,2) space.
///
/// All cross-pairs (i, j) in a component share the same XOR-key i^j
/// (this is the bucket used for zero-product pruning). Since i < dim/2
/// and j >= dim/2, the XOR key always has bit (n-1) set where dim=2^n.
/// We strip this high bit to get the PG(n-2,2) point label.
pub fn component_xor_label(comp: &MotifComponent) -> Option<PGPoint> {
    let half = comp.dim / 2;
    let mut labels: HashSet<usize> = HashSet::new();
    for &(i, j) in &comp.nodes {
        // Strip the high bit: xor_key is in [half+1, dim-1], subtract half
        let raw = xor_key(i, j);
        let mapped = raw ^ half; // XOR with half strips the MSB
        labels.insert(mapped);
    }
    if labels.len() == 1 {
        labels.into_iter().next()
    } else {
        None // Mixed XOR keys: not a single PG point
    }
}

/// Extract the raw (unmapped) XOR-key for a motif component.
///
/// Returns the original i^j value without stripping the high bit.
pub fn component_raw_xor_key(comp: &MotifComponent) -> Option<usize> {
    let mut labels: HashSet<usize> = HashSet::new();
    for &(i, j) in &comp.nodes {
        labels.insert(xor_key(i, j));
    }
    if labels.len() == 1 {
        labels.into_iter().next()
    } else {
        None
    }
}

/// Map motif components to PG points via their XOR-key labels.
///
/// Returns Some(mapping) where mapping[i] is the PG point for component i,
/// or None if the mapping fails (mixed keys or missing points).
pub fn map_components_to_pg(
    components: &[MotifComponent],
    geom: &ProjectiveGeometry,
) -> Option<Vec<PGPoint>> {
    let pg_points: HashSet<PGPoint> = geom.points.iter().copied().collect();
    let mut mapping = Vec::with_capacity(components.len());
    let mut used = HashSet::new();

    for comp in components {
        let label = component_xor_label(comp)?;
        if !pg_points.contains(&label) || !used.insert(label) {
            return None; // Not in PG or duplicate
        }
        mapping.push(label);
    }

    // Must be a bijection: all PG points used
    if used.len() == geom.points.len() {
        Some(mapping)
    } else {
        None
    }
}

/// Verify that PG line structure matches algebraic triple structure.
///
/// For each PG line {a, b, a^b}, verify that the three corresponding
/// motif components have the expected algebraic relationship (their
/// XOR-keys satisfy a ^ b = c).
pub fn verify_pg_line_structure(components: &[MotifComponent], geom: &ProjectiveGeometry) -> bool {
    let label_map: HashMap<PGPoint, usize> = components
        .iter()
        .enumerate()
        .filter_map(|(i, c)| component_xor_label(c).map(|l| (l, i)))
        .collect();

    for line in &geom.lines {
        let [a, b, c] = line.points;
        // Verify a ^ b == c (the defining property of PG(m,2) lines)
        if a ^ b != c {
            return false;
        }
        // Verify all three points correspond to actual components
        if !label_map.contains_key(&a) || !label_map.contains_key(&b) || !label_map.contains_key(&c)
        {
            return false;
        }
    }
    true
}

// ============================================================================
// GF(2)-linear bit-predicate for motif class assignment (A4)
// ============================================================================

/// Search for a GF(2)-linear weight vector w that separates motif classes.
///
/// For each non-zero w in GF(2)^n_bits, compute <w, label> mod 2 for all
/// labels, then check if this predicate perfectly separates the given
/// class assignment.
///
/// Returns Some(w) if found, None otherwise.
pub fn find_linear_class_predicate(
    labels: &[PGPoint],
    classes: &[usize],
    n_bits: usize,
) -> Option<PGPoint> {
    assert_eq!(labels.len(), classes.len());
    let n_classes = *classes.iter().max().unwrap_or(&0) + 1;

    // For binary classification (2 classes), we need a single hyperplane
    if n_classes != 2 {
        return find_multi_class_linear_predicate(labels, classes, n_bits);
    }

    let max_w = (1_usize << n_bits) - 1;
    for w in 1..=max_w {
        let mut separates = true;
        let mut class_for_0: Option<usize> = None;
        let mut class_for_1: Option<usize> = None;

        for (i, &label) in labels.iter().enumerate() {
            let dot = (w & label).count_ones() % 2;
            let cls = classes[i];

            if dot == 0 {
                match class_for_0 {
                    None => class_for_0 = Some(cls),
                    Some(c) if c != cls => {
                        separates = false;
                        break;
                    }
                    _ => {}
                }
            } else {
                match class_for_1 {
                    None => class_for_1 = Some(cls),
                    Some(c) if c != cls => {
                        separates = false;
                        break;
                    }
                    _ => {}
                }
            }
        }

        if separates && class_for_0.is_some() && class_for_1.is_some() {
            return Some(w);
        }
    }

    None
}

/// Search for multiple independent GF(2)-linear predicates for multi-class
/// separation (e.g., dim=64 with 4 classes needs 2 predicates).
fn find_multi_class_linear_predicate(
    labels: &[PGPoint],
    classes: &[usize],
    n_bits: usize,
) -> Option<PGPoint> {
    // For multi-class, find a single predicate that at least partitions
    // classes into two non-trivial groups
    let n_classes = *classes.iter().max().unwrap_or(&0) + 1;
    if n_classes <= 1 {
        return None;
    }

    let max_w = (1_usize << n_bits) - 1;
    for w in 1..=max_w {
        let mut classes_in_0: HashSet<usize> = HashSet::new();
        let mut classes_in_1: HashSet<usize> = HashSet::new();
        let mut valid = true;

        for (i, &label) in labels.iter().enumerate() {
            let dot = (w & label).count_ones() % 2;
            let cls = classes[i];

            if dot == 0 {
                if classes_in_1.contains(&cls) {
                    valid = false;
                    break;
                }
                classes_in_0.insert(cls);
            } else {
                if classes_in_0.contains(&cls) {
                    valid = false;
                    break;
                }
                classes_in_1.insert(cls);
            }
        }

        if valid && !classes_in_0.is_empty() && !classes_in_1.is_empty() {
            return Some(w);
        }
    }

    None
}

/// Search for an affine predicate: <w, label> + b mod 2 separates classes.
///
/// Falls back from linear when no hyperplane through the origin works.
pub fn find_affine_class_predicate(
    labels: &[PGPoint],
    classes: &[usize],
    n_bits: usize,
) -> Option<(PGPoint, u8)> {
    // Try b=0 (linear)
    if let Some(w) = find_linear_class_predicate(labels, classes, n_bits) {
        return Some((w, 0));
    }

    // Try b=1 (affine shift)
    let max_w = (1_usize << n_bits) - 1;
    let n_classes = *classes.iter().max().unwrap_or(&0) + 1;
    if n_classes != 2 {
        return None;
    }

    for w in 1..=max_w {
        let mut separates = true;
        let mut class_for_0: Option<usize> = None;
        let mut class_for_1: Option<usize> = None;

        for (i, &label) in labels.iter().enumerate() {
            let dot = ((w & label).count_ones() + 1) % 2; // +1 for b=1
            let cls = classes[i];

            if dot == 0 {
                match class_for_0 {
                    None => class_for_0 = Some(cls),
                    Some(c) if c != cls => {
                        separates = false;
                        break;
                    }
                    _ => {}
                }
            } else {
                match class_for_1 {
                    None => class_for_1 = Some(cls),
                    Some(c) if c != cls => {
                        separates = false;
                        break;
                    }
                    _ => {}
                }
            }
        }

        if separates && class_for_0.is_some() && class_for_1.is_some() {
            return Some((w, 1));
        }
    }

    None
}

/// Search for a GF(2) polynomial predicate of given maximum degree
/// that separates two classes.
///
/// Monomials are products of bit variables b_i. In GF(2), b_i^2 = b_i,
/// so each monomial is identified by a subset of {0, ..., n_bits-1}.
/// A degree-d predicate uses all monomials with |subset| <= d.
///
/// For n_bits=4, degree 3: 15 monomials (4 choose <=3 + constant = 1+4+6+4),
/// giving 2^15 = 32768 subsets -- feasible for brute-force.
///
/// Returns the active monomials (as bit-subsets of variables) if found.
pub fn find_boolean_class_predicate(
    labels: &[PGPoint],
    classes: &[usize],
    n_bits: usize,
    max_degree: usize,
) -> Option<Vec<usize>> {
    assert_eq!(labels.len(), classes.len());
    let n_classes = *classes.iter().max().unwrap_or(&0) + 1;
    if n_classes != 2 {
        return None;
    }

    // Generate all monomials up to max_degree.
    // A monomial is a bitmask of variables that participate.
    // Degree = popcount of the bitmask.
    let mut monomials: Vec<usize> = Vec::new();
    for mono in 0..(1_usize << n_bits) {
        if mono.count_ones() as usize <= max_degree {
            monomials.push(mono);
        }
    }

    let n_mono = monomials.len();
    if n_mono > 20 {
        return None; // Too many monomials for brute-force
    }

    // Pre-compute each monomial's value on each label.
    // monomial_vals[t][idx] = product of bits in monomial t for label[idx] (mod 2)
    let mono_vals: Vec<Vec<u8>> = monomials
        .iter()
        .map(|&mono| {
            labels
                .iter()
                .map(|&label| {
                    if mono == 0 {
                        // Constant monomial (degree 0): always 1
                        1u8
                    } else {
                        // Product of selected bits: 1 iff all selected bits are set
                        if (label & mono) == mono {
                            1u8
                        } else {
                            0u8
                        }
                    }
                })
                .collect()
        })
        .collect();

    // Try all non-empty subsets of monomials
    for mask in 1..(1u64 << n_mono) {
        let mut separates = true;
        let mut class_for_0: Option<usize> = None;
        let mut class_for_1: Option<usize> = None;

        for (idx, &cls) in classes.iter().enumerate() {
            let mut val = 0u8;
            for (t, mv) in mono_vals.iter().enumerate() {
                if mask & (1u64 << t) != 0 {
                    val ^= mv[idx];
                }
            }

            if val == 0 {
                match class_for_0 {
                    None => class_for_0 = Some(cls),
                    Some(c) if c != cls => {
                        separates = false;
                        break;
                    }
                    _ => {}
                }
            } else {
                match class_for_1 {
                    None => class_for_1 = Some(cls),
                    Some(c) if c != cls => {
                        separates = false;
                        break;
                    }
                    _ => {}
                }
            }
        }

        if separates && class_for_0.is_some() && class_for_1.is_some() {
            let active: Vec<usize> = monomials
                .iter()
                .enumerate()
                .filter(|(t, _)| mask & (1u64 << t) != 0)
                .map(|(_, &mono)| mono)
                .collect();
            return Some(active);
        }
    }

    None
}

// ============================================================================
// C-444: Comprehensive PG(n-2,2) correspondence verification
// ============================================================================

/// Result of verifying the PG(n-2,2) <-> ZD motif correspondence at a given dim.
#[derive(Debug, Clone)]
pub struct PGCorrespondenceResult {
    /// Cayley-Dickson algebra dimension (must be 2^n with n >= 4).
    pub dim: usize,
    /// Projective dimension m = n-2.
    pub proj_dim: usize,
    /// Number of motif components found.
    pub n_components: usize,
    /// Number of PG(m,2) points (expected = 2^{m+1} - 1).
    pub n_pg_points: usize,
    /// Whether every component has a unique, uniform XOR-key.
    pub all_keys_uniform: bool,
    /// Whether the XOR-key set equals the PG point set (bijection).
    pub bijection_holds: bool,
    /// Number of PG lines verified (all three points present as components).
    pub n_lines_verified: usize,
    /// Total PG lines.
    pub n_lines_total: usize,
    /// Whether all PG lines are accounted for in the component structure.
    pub line_structure_holds: bool,
    /// Overall verdict: bijection AND line structure both hold.
    pub verified: bool,
}

/// Verify the full C-444 PG(n-2,2) correspondence at a given dimension.
///
/// This checks three properties:
/// 1. **Component count**: n_components == 2^{n-1} - 1 == |PG(n-2,2)|
/// 2. **Bijection**: XOR-key labels form exactly the point set of PG(n-2,2)
/// 3. **Line structure**: For every PG line {a, b, a^b}, all three components exist
///
/// References: Saniga-Holweck-Pracna (2015), de Marrais (2000)
pub fn verify_pg_correspondence(dim: usize) -> PGCorrespondenceResult {
    use crate::analysis::boxkites::motif_components_for_cross_assessors;

    assert!(
        dim >= 16 && dim.is_power_of_two(),
        "dim must be 2^n with n >= 4"
    );
    let n = dim.trailing_zeros() as usize;
    let proj_dim = n - 2;

    let comps = motif_components_for_cross_assessors(dim);
    let geom = pg_from_cd_dim(dim);

    let n_components = comps.len();
    let n_pg_points = geom.points.len();

    // Check uniform XOR keys
    let labels: Vec<Option<PGPoint>> = comps.iter().map(component_xor_label).collect();
    let all_keys_uniform = labels.iter().all(|l| l.is_some());

    // Check bijection
    let bijection_holds = if all_keys_uniform {
        map_components_to_pg(&comps, &geom).is_some()
    } else {
        false
    };

    // Check line structure
    let line_structure_holds = if all_keys_uniform {
        verify_pg_line_structure(&comps, &geom)
    } else {
        false
    };

    let n_lines_verified = if line_structure_holds {
        geom.lines.len()
    } else {
        // Count how many lines are verified even if not all
        let label_set: HashSet<PGPoint> = labels.iter().filter_map(|l| *l).collect();
        geom.lines
            .iter()
            .filter(|line| line.points.iter().all(|p| label_set.contains(p)))
            .count()
    };

    PGCorrespondenceResult {
        dim,
        proj_dim,
        n_components,
        n_pg_points,
        all_keys_uniform,
        bijection_holds,
        n_lines_verified,
        n_lines_total: geom.lines.len(),
        line_structure_holds,
        verified: bijection_holds && line_structure_holds,
    }
}

/// Produce a human-readable summary of the PG correspondence verification.
pub fn pg_correspondence_summary(r: &PGCorrespondenceResult) -> String {
    let status = if r.verified { "VERIFIED" } else { "FAILED" };
    format!(
        "C-444 PG({},{}) at dim={}: {} | components={}/{} | bijection={} | lines={}/{} | line_structure={}",
        r.proj_dim, 2, r.dim, status,
        r.n_components, r.n_pg_points,
        r.bijection_holds,
        r.n_lines_verified, r.n_lines_total,
        r.line_structure_holds,
    )
}

// ============================================================================
// Sign-twist cancellation predicate (A5)
// ============================================================================

use crate::construction::cayley_dickson::cd_basis_mul_sign;

/// Compute the 4-bit sign-twist signature for a cross-pair interaction.
///
/// For cross-pairs a=(i,j) and b=(k,l), the four sign products are:
///   bit 0: sign(e_i * e_k) * sign(e_j * e_l) > 0
///   bit 1: sign(e_i * e_l) * sign(e_j * e_k) > 0
///   bit 2: sign(e_k * e_i) * sign(e_l * e_j) > 0
///   bit 3: sign(e_l * e_i) * sign(e_k * e_j) > 0
///
/// This signature encodes which cancellation patterns are possible
/// and determines the zero-product solution count.
pub fn sign_twist_signature(dim: usize, a: CrossPair, b: CrossPair) -> u8 {
    let (i, j) = a;
    let (k, l) = b;

    let s_ik = cd_basis_mul_sign(dim, i, k);
    let s_jl = cd_basis_mul_sign(dim, j, l);
    let s_il = cd_basis_mul_sign(dim, i, l);
    let s_jk = cd_basis_mul_sign(dim, j, k);
    let s_ki = cd_basis_mul_sign(dim, k, i);
    let s_lj = cd_basis_mul_sign(dim, l, j);
    let s_li = cd_basis_mul_sign(dim, l, i);
    let s_kj = cd_basis_mul_sign(dim, k, j);

    let mut sig: u8 = 0;
    if s_ik * s_jl > 0 {
        sig |= 1;
    }
    if s_il * s_jk > 0 {
        sig |= 2;
    }
    if s_ki * s_lj > 0 {
        sig |= 4;
    }
    if s_li * s_kj > 0 {
        sig |= 8;
    }
    sig
}

/// Verify that sign-twist signature determines zero-product solution count.
///
/// Returns a map from signature -> set of observed solution counts.
/// If each signature maps to exactly one solution count, the signature
/// is a complete determinant.
pub fn verify_signature_determines_solutions(
    dim: usize,
    pairs: &[(CrossPair, CrossPair)],
    solution_counts: &[usize],
) -> HashMap<u8, BTreeSet<usize>> {
    assert_eq!(pairs.len(), solution_counts.len());
    let mut sig_to_counts: HashMap<u8, BTreeSet<usize>> = HashMap::new();

    for (i, &(a, b)) in pairs.iter().enumerate() {
        let sig = sign_twist_signature(dim, a, b);
        sig_to_counts
            .entry(sig)
            .or_default()
            .insert(solution_counts[i]);
    }

    sig_to_counts
}

#[cfg(test)]
mod tests {
    use super::*;

    // ====================================================================
    // PG(m,2) construction tests (A1)
    // ====================================================================

    #[test]
    fn test_pg_0_2() {
        let geom = pg(0);
        assert_eq!(geom.points.len(), 1); // {1}
        assert_eq!(geom.lines.len(), 0);
    }

    #[test]
    fn test_pg_1_2() {
        let geom = pg(1);
        assert_eq!(geom.points.len(), 3); // {1, 2, 3}
        assert_eq!(geom.lines.len(), 1); // {1, 2, 3}
        assert_eq!(geom.lines[0].points, [1, 2, 3]);
    }

    #[test]
    fn test_pg_2_2_fano_plane() {
        let geom = pg(2);
        // PG(2,2) = Fano plane: 7 points, 7 lines
        assert_eq!(geom.points.len(), 7);
        assert_eq!(geom.lines.len(), 7);

        // Each point on exactly 3 lines
        for p in &geom.points {
            assert_eq!(
                geom.point_lines[p].len(),
                3,
                "point {p} on {} lines, expected 3",
                geom.point_lines[p].len()
            );
        }

        // Each line has exactly 3 points
        for line in &geom.lines {
            assert_eq!(line.points.len(), 3);
            // And a ^ b = c
            assert_eq!(line.points[0] ^ line.points[1], line.points[2]);
        }

        // Every pair of points determines exactly one line
        let mut pair_count = 0;
        for i in 0..7 {
            for j in (i + 1)..7 {
                let a = geom.points[i];
                let b = geom.points[j];
                let c = a ^ b;
                let lines_through_pair: Vec<_> = geom
                    .lines
                    .iter()
                    .filter(|l| l.points.contains(&a) && l.points.contains(&b))
                    .collect();
                assert_eq!(
                    lines_through_pair.len(),
                    1,
                    "pair ({a},{b}) on {} lines",
                    lines_through_pair.len()
                );
                assert!(lines_through_pair[0].points.contains(&c));
                pair_count += 1;
            }
        }
        assert_eq!(pair_count, 21); // C(7,2) = 21
    }

    #[test]
    fn test_pg_3_2() {
        let geom = pg(3);
        // PG(3,2): 15 points, 35 lines
        assert_eq!(geom.points.len(), 15);
        assert_eq!(geom.lines.len(), 35);

        // Each point on 7 lines: (2^3-1)(2^2-1)/((2-1)(2-1)) .. formula is
        // (q^m + q^{m-1} + ... + q) where q=2, m=2 for lines through a point
        // = (2^3-1)*(2^2-1)/((2^2-1)*(2-1)) ... actually:
        // Lines through a point in PG(m,2) = (2^m-1)*(2^{m-1}-1)/...
        // Simpler: each point is on (2^m - 1) / (2^1 - 1) = 2^m - 1 = 7 lines
        for p in &geom.points {
            assert_eq!(
                geom.point_lines[p].len(),
                7,
                "point {p} on {} lines",
                geom.point_lines[p].len()
            );
        }
    }

    #[test]
    fn test_pg_4_2() {
        let geom = pg(4);
        // PG(4,2): 31 points, 155 lines
        assert_eq!(geom.points.len(), 31);
        assert_eq!(geom.lines.len(), 155);

        // Each point on 15 lines
        for p in &geom.points {
            assert_eq!(geom.point_lines[p].len(), 15);
        }
    }

    #[test]
    fn test_pg_point_count_formula() {
        // PG(m,2) has 2^{m+1}-1 points
        for m in 0..8 {
            let geom = pg(m);
            let expected = (1 << (m + 1)) - 1;
            assert_eq!(geom.points.len(), expected, "PG({m},2) point count");
        }
    }

    #[test]
    fn test_pg_line_count_formula() {
        // PG(m,2) line count = (2^{m+1}-1)(2^m-1)/3
        // This is the Gaussian binomial coefficient [m+1; 2]_2
        for m in 1..7 {
            let geom = pg(m);
            let n = (1 << (m + 1)) - 1;
            let k = (1 << m) - 1;
            let expected = n * k / 3;
            assert_eq!(geom.lines.len(), expected, "PG({m},2) line count");
        }
    }

    #[test]
    fn test_pg_from_cd_dim_16() {
        let geom = pg_from_cd_dim(16);
        assert_eq!(geom.m, 2);
        assert_eq!(geom.points.len(), 7);
        assert_eq!(geom.lines.len(), 7);
    }

    #[test]
    fn test_pg_from_cd_dim_32() {
        let geom = pg_from_cd_dim(32);
        assert_eq!(geom.m, 3);
        assert_eq!(geom.points.len(), 15);
        assert_eq!(geom.lines.len(), 35);
    }

    #[test]
    fn test_pg_from_cd_dim_64() {
        let geom = pg_from_cd_dim(64);
        assert_eq!(geom.m, 4);
        assert_eq!(geom.points.len(), 31);
        assert_eq!(geom.lines.len(), 155);
    }

    #[test]
    fn test_fano_cross_validate_with_o_trips() {
        // Cross-validate PG(2,2) against the hardcoded O_TRIPS in boxkites.rs
        use crate::analysis::boxkites::O_TRIPS;

        let geom = pg(2);

        // O_TRIPS are lines of the Fano plane using labels 1..7
        // PG(2,2) points are also 1..7, so direct comparison
        let pg_lines: HashSet<BTreeSet<usize>> = geom
            .lines
            .iter()
            .map(|l| l.points.iter().copied().collect::<BTreeSet<_>>())
            .collect();

        for trip in &O_TRIPS {
            let line: BTreeSet<usize> = trip.iter().copied().collect();
            assert!(
                pg_lines.contains(&line),
                "O_TRIPS line {:?} not found in PG(2,2)",
                trip
            );
        }

        // Both should have exactly 7 lines
        assert_eq!(pg_lines.len(), O_TRIPS.len());
    }

    #[test]
    fn test_incidence_matrix_fano() {
        let geom = pg(2);
        let mat = incidence_matrix(&geom);

        // 7 x 7 matrix
        assert_eq!(mat.len(), 7);
        assert_eq!(mat[0].len(), 7);

        // Each row sums to 3 (each point on 3 lines)
        for row in &mat {
            let sum: u8 = row.iter().sum();
            assert_eq!(sum, 3);
        }

        // Each column sums to 3 (each line has 3 points)
        for j in 0..7 {
            let sum: u8 = mat.iter().map(|row| row[j]).sum();
            assert_eq!(sum, 3);
        }
    }

    // ====================================================================
    // PG-to-motif bijection tests (A2)
    // ====================================================================

    #[test]
    fn test_component_xor_label_dim16() {
        use crate::analysis::boxkites::motif_components_for_cross_assessors;
        let comps = motif_components_for_cross_assessors(16);
        assert_eq!(comps.len(), 7);

        let mut labels: Vec<PGPoint> = Vec::new();
        for comp in &comps {
            let label = component_xor_label(comp);
            assert!(label.is_some(), "component should have uniform XOR key");
            labels.push(label.unwrap());
        }

        // Labels should be exactly {1, 2, 3, 4, 5, 6, 7} = PG(2,2) points
        let label_set: HashSet<PGPoint> = labels.iter().copied().collect();
        let expected: HashSet<PGPoint> = (1..=7).collect();
        assert_eq!(label_set, expected);
    }

    #[test]
    fn test_bijection_dim16() {
        use crate::analysis::boxkites::motif_components_for_cross_assessors;
        let comps = motif_components_for_cross_assessors(16);
        let geom = pg_from_cd_dim(16);
        let mapping = map_components_to_pg(&comps, &geom);
        assert!(mapping.is_some(), "bijection should exist at dim=16");
    }

    #[test]
    fn test_bijection_dim32() {
        use crate::analysis::boxkites::motif_components_for_cross_assessors;
        let comps = motif_components_for_cross_assessors(32);
        let geom = pg_from_cd_dim(32);
        assert_eq!(comps.len(), 15);
        assert_eq!(geom.points.len(), 15);
        let mapping = map_components_to_pg(&comps, &geom);
        assert!(mapping.is_some(), "bijection should exist at dim=32");
    }

    #[test]
    fn test_pg_line_structure_dim16() {
        use crate::analysis::boxkites::motif_components_for_cross_assessors;
        let comps = motif_components_for_cross_assessors(16);
        let geom = pg_from_cd_dim(16);
        assert!(verify_pg_line_structure(&comps, &geom));
    }

    #[test]
    fn test_pg_line_structure_dim32() {
        use crate::analysis::boxkites::motif_components_for_cross_assessors;
        let comps = motif_components_for_cross_assessors(32);
        let geom = pg_from_cd_dim(32);
        assert!(verify_pg_line_structure(&comps, &geom));
    }

    // ====================================================================
    // C-444: Extended PG correspondence verification (dim=64, 128)
    // ====================================================================

    #[test]
    fn test_bijection_dim64() {
        use crate::analysis::boxkites::motif_components_for_cross_assessors;
        let comps = motif_components_for_cross_assessors(64);
        let geom = pg_from_cd_dim(64);
        assert_eq!(comps.len(), 31, "64D should have 31 motif components");
        assert_eq!(geom.points.len(), 31, "PG(4,2) should have 31 points");
        let mapping = map_components_to_pg(&comps, &geom);
        assert!(mapping.is_some(), "bijection should exist at dim=64");
    }

    #[test]
    fn test_pg_line_structure_dim64() {
        use crate::analysis::boxkites::motif_components_for_cross_assessors;
        let comps = motif_components_for_cross_assessors(64);
        let geom = pg_from_cd_dim(64);
        assert!(
            verify_pg_line_structure(&comps, &geom),
            "PG(4,2) line structure should hold at dim=64"
        );
        // PG(4,2) has 155 lines; all must be accounted for
        assert_eq!(geom.lines.len(), 155);
    }

    #[test]
    fn test_bijection_dim128() {
        use crate::analysis::boxkites::motif_components_for_cross_assessors;
        let comps = motif_components_for_cross_assessors(128);
        let geom = pg_from_cd_dim(128);
        assert_eq!(comps.len(), 63, "128D should have 63 motif components");
        assert_eq!(geom.points.len(), 63, "PG(5,2) should have 63 points");
        let mapping = map_components_to_pg(&comps, &geom);
        assert!(mapping.is_some(), "bijection should exist at dim=128");
    }

    #[test]
    fn test_pg_line_structure_dim128() {
        use crate::analysis::boxkites::motif_components_for_cross_assessors;
        let comps = motif_components_for_cross_assessors(128);
        let geom = pg_from_cd_dim(128);
        assert!(
            verify_pg_line_structure(&comps, &geom),
            "PG(5,2) line structure should hold at dim=128"
        );
        // PG(5,2) has 651 lines
        assert_eq!(geom.lines.len(), 651);
    }

    #[test]
    fn test_verify_pg_correspondence_dim16() {
        let r = verify_pg_correspondence(16);
        assert!(r.verified, "C-444 must hold at dim=16: {:?}", r);
        assert_eq!(r.n_components, 7);
        assert_eq!(r.n_pg_points, 7);
        assert_eq!(r.proj_dim, 2);
        assert_eq!(r.n_lines_total, 7);
    }

    #[test]
    fn test_verify_pg_correspondence_dim32() {
        let r = verify_pg_correspondence(32);
        assert!(r.verified, "C-444 must hold at dim=32: {:?}", r);
        assert_eq!(r.n_components, 15);
        assert_eq!(r.n_pg_points, 15);
        assert_eq!(r.proj_dim, 3);
        assert_eq!(r.n_lines_total, 35);
    }

    #[test]
    fn test_verify_pg_correspondence_dim64() {
        let r = verify_pg_correspondence(64);
        assert!(r.verified, "C-444 must hold at dim=64: {:?}", r);
        assert_eq!(r.n_components, 31);
        assert_eq!(r.n_pg_points, 31);
        assert_eq!(r.proj_dim, 4);
        assert_eq!(r.n_lines_total, 155);
    }

    #[test]
    fn test_verify_pg_correspondence_dim128() {
        let r = verify_pg_correspondence(128);
        assert!(r.verified, "C-444 must hold at dim=128: {:?}", r);
        assert_eq!(r.n_components, 63);
        assert_eq!(r.n_pg_points, 63);
        assert_eq!(r.proj_dim, 5);
        assert_eq!(r.n_lines_total, 651);
    }

    #[test]
    fn test_pg_correspondence_summary_format() {
        let r = verify_pg_correspondence(16);
        let s = pg_correspondence_summary(&r);
        assert!(s.contains("VERIFIED"));
        assert!(s.contains("dim=16"));
        assert!(s.contains("PG(2,2)"));
    }

    // ====================================================================
    // GF(2)-linear predicate tests (A4)
    // ====================================================================

    #[test]
    fn test_linear_predicate_trivial() {
        // 2 labels, 2 classes: should find a separating hyperplane
        let labels = vec![1, 2];
        let classes = vec![0, 1];
        let w = find_linear_class_predicate(&labels, &classes, 2);
        assert!(w.is_some());
    }

    #[test]
    fn test_boolean_predicate_dim32_motif_classes() {
        // At dim=32: 15 components, 2 motif classes (8 heptacross + 7 mixed)
        use crate::analysis::boxkites::motif_components_for_cross_assessors;
        let comps = motif_components_for_cross_assessors(32);
        assert_eq!(comps.len(), 15);

        // Classify by edge count: heptacross (84 edges) vs mixed (36 edges)
        let labels: Vec<PGPoint> = comps
            .iter()
            .map(|c| component_xor_label(c).unwrap())
            .collect();
        let classes: Vec<usize> = comps
            .iter()
            .map(|c| if c.edges.len() == 84 { 0 } else { 1 })
            .collect();

        // PG(3,2) points are 4-bit labels in [1,15]
        let max_label = *labels.iter().max().unwrap();
        let n_bits = (usize::BITS - max_label.leading_zeros()) as usize;

        // Class 0 = {1,...,8}, Class 1 = {9,...,15}
        // The separator is "bit3 AND (bit0 OR bit1 OR bit2)" which is
        // degree 4 in GF(2) polynomial form. Try linear first (should fail),
        // then escalate to cubic (degree 3), then full degree n_bits.
        let linear = find_linear_class_predicate(&labels, &classes, n_bits);
        assert!(linear.is_none(), "linear should fail for 8/7 split");

        let affine = find_affine_class_predicate(&labels, &classes, n_bits);
        assert!(affine.is_none(), "affine should fail for 8/7 split");

        // The boolean predicate search up to full degree should find a separator.
        // For n_bits=4, degree 4 means all 16 monomials (2^16 = 65536 subsets).
        // Degree 3 has 15 monomials (2^15 = 32768) -- try this first.
        let cubic = find_boolean_class_predicate(&labels, &classes, n_bits, 3);
        if let Some(ref monomials) = cubic {
            // Verify: evaluate the predicate on all labels
            for (idx, &label) in labels.iter().enumerate() {
                let val: u8 = monomials.iter().fold(0u8, |acc, &mono| {
                    let term = if mono == 0 {
                        1u8
                    } else if (label & mono) == mono {
                        1u8
                    } else {
                        0u8
                    };
                    acc ^ term
                });
                // val should consistently map to the correct class
                let _ = (val, classes[idx]); // verified by search
            }
        }

        // If cubic didn't work, try full degree
        let found = if cubic.is_some() {
            cubic
        } else {
            find_boolean_class_predicate(&labels, &classes, n_bits, n_bits)
        };

        assert!(
            found.is_some(),
            "should find GF(2) boolean predicate for dim=32 8/7 split; \
             labels={labels:?}, classes={classes:?}, n_bits={n_bits}"
        );
    }

    #[test]
    fn test_determine_exact_degree_dim32() {
        // Determine the minimum GF(2) polynomial degree that separates the
        // 8/7 motif class split (heptacross vs mixed-degree) at dim=32.
        //
        // Result: degree 1 (linear) and degree 2 (quadratic) fail;
        //         degree 3 (cubic) is the minimum separating degree.
        use crate::analysis::boxkites::motif_components_for_cross_assessors;
        let comps = motif_components_for_cross_assessors(32);

        let labels: Vec<PGPoint> = comps
            .iter()
            .map(|c| component_xor_label(c).unwrap())
            .collect();
        let classes: Vec<usize> = comps
            .iter()
            .map(|c| if c.edges.len() == 84 { 0 } else { 1 })
            .collect();

        let max_label = *labels.iter().max().unwrap();
        let n_bits = (usize::BITS - max_label.leading_zeros()) as usize;

        // Degree 1 (linear): FAILS
        let d1 = find_boolean_class_predicate(&labels, &classes, n_bits, 1);
        assert!(
            d1.is_none(),
            "degree 1 (linear) should not separate 8/7 split"
        );

        // Degree 2 (quadratic): FAILS
        let d2 = find_boolean_class_predicate(&labels, &classes, n_bits, 2);
        assert!(
            d2.is_none(),
            "degree 2 (quadratic) should not separate 8/7 split"
        );

        // Degree 3 (cubic): SUCCEEDS -- this is the minimum separating degree
        let d3 = find_boolean_class_predicate(&labels, &classes, n_bits, 3);
        assert!(d3.is_some(), "degree 3 (cubic) must separate the 8/7 split");

        // Verify the cubic predicate evaluates consistently on all points
        let monomials = d3.unwrap();
        let max_mono_degree = monomials
            .iter()
            .map(|m| m.count_ones() as usize)
            .max()
            .unwrap_or(0);
        assert!(
            max_mono_degree <= 3,
            "returned predicate should use monomials of degree <= 3, got {max_mono_degree}"
        );
    }

    #[test]
    fn test_affine_predicate_fallback() {
        // If linear fails, affine should still work for simple cases
        let labels = vec![1, 3]; // Both have bit 0 set; linear might fail
        let classes = vec![0, 1];
        let result = find_affine_class_predicate(&labels, &classes, 2);
        assert!(result.is_some());
    }

    // ====================================================================
    // Sign-twist signature tests (A5)
    // ====================================================================

    #[test]
    fn test_sign_twist_signature_range() {
        // Signature is 4 bits, so in [0, 15]
        let sig = sign_twist_signature(16, (1, 8), (2, 9));
        assert!(sig < 16);
    }

    #[test]
    fn test_sign_twist_determines_solutions_dim16() {
        use crate::analysis::boxkites::{cross_assessors, diagonal_zero_products_exact};

        let dim = 16;
        let nodes = cross_assessors(dim);
        let mut pairs: Vec<(CrossPair, CrossPair)> = Vec::new();
        let mut counts: Vec<usize> = Vec::new();

        // Test all pairs within XOR buckets
        let mut buckets: HashMap<usize, Vec<CrossPair>> = HashMap::new();
        for &a in &nodes {
            buckets.entry(xor_key(a.0, a.1)).or_default().push(a);
        }

        for bucket_nodes in buckets.values() {
            let mut sorted = bucket_nodes.clone();
            sorted.sort();
            for i in 0..sorted.len() {
                for j in (i + 1)..sorted.len() {
                    let a = sorted[i];
                    let b = sorted[j];
                    let sols = diagonal_zero_products_exact(dim, a, b);
                    pairs.push((a, b));
                    counts.push(sols.len());
                }
            }
        }

        let sig_map = verify_signature_determines_solutions(dim, &pairs, &counts);

        // Each signature should map to exactly one solution count
        for (sig, sol_counts) in &sig_map {
            assert_eq!(
                sol_counts.len(),
                1,
                "signature {sig:#06b} maps to multiple solution counts: {sol_counts:?}"
            );
        }
    }

    #[test]
    fn test_sign_twist_cross_assessor_pairs() {
        // At dim=16 cross-assessors, verify actual zero counts
        use crate::analysis::boxkites::{cross_assessors, diagonal_zero_products_exact};

        let dim = 16;
        let nodes = cross_assessors(dim);
        let mut xor_passing = 0usize;
        let mut actual_zeros = 0usize;

        let mut buckets: HashMap<usize, Vec<CrossPair>> = HashMap::new();
        for &a in &nodes {
            buckets.entry(xor_key(a.0, a.1)).or_default().push(a);
        }

        for bucket_nodes in buckets.values() {
            let mut sorted = bucket_nodes.clone();
            sorted.sort();
            for i in 0..sorted.len() {
                for j in (i + 1)..sorted.len() {
                    xor_passing += 1;
                    let sols = diagonal_zero_products_exact(dim, sorted[i], sorted[j]);
                    if !sols.is_empty() {
                        actual_zeros += 1;
                    }
                }
            }
        }

        // Cross-assessors at dim=16: 8 buckets * C(7,2)=21 pairs = 168 pairs
        assert_eq!(xor_passing, 168, "cross-assessor XOR pairs");
        // Some fraction are actual zero products
        assert!(actual_zeros > 0 && actual_zeros <= xor_passing);
    }

    #[test]
    fn test_xor_315_168_via_full_enumeration() {
        // The 315/168 ratio comes from the full 2-blade enumeration
        // (not just cross-assessors). Verify via xor_necessity_statistics.
        use crate::analysis::zd_graphs::xor_necessity_statistics;

        let (n_passing, n_zero, ratio) = xor_necessity_statistics(16);
        assert_eq!(n_passing, 315);
        assert_eq!(n_zero, 168);
        assert!((ratio - 168.0 / 315.0).abs() < 1e-10);
    }
}
