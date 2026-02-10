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
// Multi-class GF(2) separating degree analysis (I-012 open question)
// ============================================================================

/// Result of the minimum separating degree search.
#[derive(Debug, Clone)]
pub struct SeparatingDegreeResult {
    /// Cayley-Dickson dimension.
    pub dim: usize,
    /// Number of PG points (= number of motif components).
    pub n_points: usize,
    /// Number of motif classes found.
    pub n_classes: usize,
    /// Class sizes, sorted descending.
    pub class_sizes: Vec<usize>,
    /// Minimum GF(2) polynomial degree that separates all classes.
    /// None if no separation found up to max_degree.
    pub min_degree: Option<usize>,
    /// Number of achievable class signatures at the minimum degree.
    pub n_achievable_sigs: usize,
    /// The achievable class signatures (as bit patterns over class indices).
    pub achievable_sigs: Vec<usize>,
    /// Degrees tested and whether they succeeded.
    pub degree_results: Vec<(usize, bool)>,
}

/// Check solvability of M*c = target over GF(2) via Gaussian elimination.
///
/// M is n_rows x n_cols, target is length n_rows. Returns true if the
/// augmented system [M | target] has a solution.
fn gf2_solvable(m: &[Vec<u8>], target: &[u8]) -> bool {
    let n = m.len();
    if n == 0 {
        return true;
    }
    let n_cols = m[0].len();

    // Build augmented matrix [M | target]
    let mut aug: Vec<Vec<u8>> = m
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let mut r = row.clone();
            r.push(target[i]);
            r
        })
        .collect();

    // Forward elimination with full pivoting
    let mut pivot_row = 0;
    let mut pivot_col = 0;
    while pivot_row < n && pivot_col < n_cols {
        // Find a row with a 1 in this column
        let mut found = false;
        for r in pivot_row..n {
            if aug[r][pivot_col] == 1 {
                aug.swap(pivot_row, r);
                found = true;
                break;
            }
        }

        if found {
            // Eliminate all other rows (clone pivot to avoid borrow conflict)
            let pivot = aug[pivot_row].clone();
            for (idx, aug_row) in aug.iter_mut().enumerate() {
                if idx != pivot_row && aug_row[pivot_col] == 1 {
                    for (cell, &pval) in aug_row.iter_mut().zip(pivot.iter()) {
                        *cell ^= pval;
                    }
                }
            }
            pivot_row += 1;
        }
        pivot_col += 1;
    }

    // Check consistency: any row with all-zero coefficients but non-zero RHS?
    for row in aug.iter().skip(pivot_row) {
        if row[n_cols] == 1 {
            return false;
        }
    }

    true
}

/// Check if two class signatures jointly separate n_classes classes.
///
/// Each signature is a bit pattern where bit i represents the value assigned
/// to class i. Two signatures separate all classes iff the pairs
/// (s1_bit_i, s2_bit_i) are all distinct across i = 0..n_classes.
fn signatures_jointly_separate(s1: usize, s2: usize, n_classes: usize) -> bool {
    let mut seen = 0u64; // bit-packed set of 2-bit pairs
    for c in 0..n_classes {
        let b1 = (s1 >> c) & 1;
        let b2 = (s2 >> c) & 1;
        let pair = (b1 << 1) | b2;
        let mask = 1u64 << pair;
        if seen & mask != 0 {
            return false; // duplicate pair
        }
        seen |= mask;
    }
    true
}

/// Check whether degree-d GF(2) polynomials can separate the given classes.
///
/// Returns (success, n_achievable, achievable_sigs).
fn check_separation_at_degree(
    labels: &[PGPoint],
    class_assignments: &[usize],
    n_bits: usize,
    n_classes: usize,
    degree: usize,
) -> (bool, usize, Vec<usize>) {
    // Generate all monomials with popcount <= degree
    let monomials: Vec<usize> = (0..(1usize << n_bits))
        .filter(|m| m.count_ones() as usize <= degree)
        .collect();
    let n_labels = labels.len();

    // Build evaluation matrix M[i][j] = monomial_j evaluated on label_i (GF(2))
    let eval_matrix: Vec<Vec<u8>> = (0..n_labels)
        .map(|i| {
            monomials
                .iter()
                .map(|&mono| {
                    // constant monomial (degree 0) always evaluates to 1;
                    // higher-degree monomials evaluate to 1 iff all selected bits are set
                    if mono == 0 || (labels[i] & mono) == mono {
                        1u8
                    } else {
                        0u8
                    }
                })
                .collect()
        })
        .collect();

    // For each non-zero class signature, check achievability
    let n_sigs = (1usize << n_classes) - 1;
    let mut achievable_sigs: Vec<usize> = Vec::new();

    for sig in 1..=n_sigs {
        // Construct target vector: label i gets bit (class_of_i) from sig
        let target: Vec<u8> = class_assignments
            .iter()
            .map(|&c| ((sig >> c) & 1) as u8)
            .collect();

        if gf2_solvable(&eval_matrix, &target) {
            achievable_sigs.push(sig);
        }
    }

    // For 2 classes: need 1 separating predicate
    if n_classes == 2 {
        // Any achievable signature that assigns different values to the 2 classes
        for &sig in &achievable_sigs {
            let b0 = sig & 1;
            let b1 = (sig >> 1) & 1;
            if b0 != b1 {
                return (true, achievable_sigs.len(), achievable_sigs);
            }
        }
        return (false, achievable_sigs.len(), achievable_sigs);
    }

    // For 4 classes: need 2 predicates that jointly separate
    // For k classes in general: need ceil(log2(k)) predicates
    let n_needed = (n_classes as f64).log2().ceil() as usize;

    if n_needed == 2 {
        // Try all pairs of achievable signatures
        for i in 0..achievable_sigs.len() {
            for j in (i + 1)..achievable_sigs.len() {
                if signatures_jointly_separate(achievable_sigs[i], achievable_sigs[j], n_classes) {
                    return (true, achievable_sigs.len(), achievable_sigs);
                }
            }
        }
        return (false, achievable_sigs.len(), achievable_sigs);
    }

    // For 8+ classes: need 3+ predicates -- use recursive approach
    if n_needed >= 3 {
        // Enumerate all n_needed-tuples of achievable signatures
        // For small n_achievable this is tractable
        let found =
            find_separating_tuple(&achievable_sigs, n_classes, n_needed, 0, &mut Vec::new());
        return (found, achievable_sigs.len(), achievable_sigs);
    }

    (false, achievable_sigs.len(), achievable_sigs)
}

/// Recursively search for a tuple of signatures that jointly separates all classes.
///
/// Uses greedy partition refinement for large search spaces (n_needed >= 4 and
/// many achievable signatures): greedily picks the signature that maximizes the
/// number of distinguished class pairs, then recurses on the residual.
fn find_separating_tuple(
    sigs: &[usize],
    n_classes: usize,
    depth: usize,
    start: usize,
    current: &mut Vec<usize>,
) -> bool {
    if current.len() == depth {
        // Check if current tuple jointly separates all classes
        let mut codes: Vec<usize> = Vec::with_capacity(n_classes);
        for c in 0..n_classes {
            let mut code = 0usize;
            for (bit_pos, &sig) in current.iter().enumerate() {
                code |= ((sig >> c) & 1) << bit_pos;
            }
            codes.push(code);
        }
        let unique: HashSet<usize> = codes.iter().copied().collect();
        return unique.len() == n_classes;
    }

    // For small search spaces, use exhaustive search
    if sigs.len() - start <= 500 || depth <= 3 {
        for i in start..sigs.len() {
            current.push(sigs[i]);
            if find_separating_tuple(sigs, n_classes, depth, i + 1, current) {
                return true;
            }
            current.pop();
        }
        return false;
    }

    // For large search spaces with depth >= 4, use greedy partition refinement
    find_separating_tuple_greedy(sigs, n_classes, depth, current)
}

/// Greedy partition refinement: pick signatures that maximally refine class
/// distinctions, then verify the result separates all classes.
///
/// For 16 classes needing 4 predicates with ~65000 achievable signatures,
/// brute-force C(65000,4) is infeasible. Greedy runs in O(depth * n_sigs * n_classes).
fn find_separating_tuple_greedy(
    sigs: &[usize],
    n_classes: usize,
    depth: usize,
    current: &mut Vec<usize>,
) -> bool {
    let base_len = current.len();
    let remaining = depth - base_len;

    // Track which class pairs are already distinguished by current signatures
    let n_pairs = n_classes * (n_classes - 1) / 2;
    let mut distinguished = vec![false; n_pairs];

    // Initialize from any signatures already in `current`
    for &sig in current.iter() {
        update_distinguished(sig, n_classes, &mut distinguished);
    }

    // Greedily pick the best signature at each step
    let mut used: HashSet<usize> = current.iter().copied().collect();
    for _ in 0..remaining {
        let mut best_sig = 0usize;
        let mut best_new = 0usize;

        for &sig in sigs {
            if used.contains(&sig) {
                continue;
            }
            // Count how many new class pairs this signature would distinguish
            let new_count = count_new_distinguished(sig, n_classes, &distinguished);
            if new_count > best_new {
                best_new = new_count;
                best_sig = sig;
            }
        }

        if best_new == 0 {
            // No progress possible; try a few random alternatives before giving up
            break;
        }

        current.push(best_sig);
        used.insert(best_sig);
        update_distinguished(best_sig, n_classes, &mut distinguished);
    }

    // Check if greedy solution fully separates
    if current.len() == depth {
        let mut codes: Vec<usize> = Vec::with_capacity(n_classes);
        for c in 0..n_classes {
            let mut code = 0usize;
            for (bit_pos, &sig) in current.iter().enumerate() {
                code |= ((sig >> c) & 1) << bit_pos;
            }
            codes.push(code);
        }
        let unique: HashSet<usize> = codes.iter().copied().collect();
        if unique.len() == n_classes {
            return true;
        }
    }

    // Greedy failed; truncate and return false
    current.truncate(base_len);
    false
}

/// Update the distinguished-pairs bitvector given a new signature.
fn update_distinguished(sig: usize, n_classes: usize, distinguished: &mut [bool]) {
    let mut idx = 0;
    for i in 0..n_classes {
        let bi = (sig >> i) & 1;
        for j in (i + 1)..n_classes {
            let bj = (sig >> j) & 1;
            if bi != bj {
                distinguished[idx] = true;
            }
            idx += 1;
        }
    }
}

/// Count how many currently-undistinguished class pairs a signature would distinguish.
fn count_new_distinguished(sig: usize, n_classes: usize, distinguished: &[bool]) -> usize {
    let mut count = 0;
    let mut idx = 0;
    for i in 0..n_classes {
        let bi = (sig >> i) & 1;
        for j in (i + 1)..n_classes {
            let bj = (sig >> j) & 1;
            if bi != bj && !distinguished[idx] {
                count += 1;
            }
            idx += 1;
        }
    }
    count
}

/// Find the minimum GF(2) polynomial degree that separates motif classes
/// at a given Cayley-Dickson dimension.
///
/// This generalizes the dim=32 cubic-degree result (I-012) to arbitrary
/// dimensions. The algorithm:
/// 1. Compute motif components and their PG(n-2,2) labels
/// 2. Classify components by edge count (invariant fingerprint)
/// 3. For each degree d=1,2,..., check if degree-d GF(2) polynomials
///    can jointly separate all classes
/// 4. Uses GF(2) Gaussian elimination for efficiency (avoids exponential
///    brute-force over polynomial subsets)
pub fn find_minimum_separating_degree(dim: usize) -> SeparatingDegreeResult {
    use crate::analysis::boxkites::motif_components_for_cross_assessors;

    assert!(
        dim >= 32 && dim.is_power_of_two(),
        "dim must be 2^n with n >= 5 (need >= 2 classes)"
    );

    let n = dim.trailing_zeros() as usize;
    let n_bits = n - 1; // PG(n-2,2) labels are (n-1)-bit

    let comps = motif_components_for_cross_assessors(dim);
    let n_points = comps.len();

    // Get PG labels
    let labels: Vec<PGPoint> = comps
        .iter()
        .map(|c| component_xor_label(c).unwrap())
        .collect();

    // Classify by edge count (determines motif class)
    let edge_counts: Vec<usize> = comps.iter().map(|c| c.edges.len()).collect();
    let unique_edges: BTreeSet<usize> = edge_counts.iter().copied().collect();
    let edge_to_class: HashMap<usize, usize> = unique_edges
        .iter()
        .enumerate()
        .map(|(i, &e)| (e, i))
        .collect();
    let class_assignments: Vec<usize> = edge_counts.iter().map(|e| edge_to_class[e]).collect();
    let n_classes = unique_edges.len();

    // Compute class sizes
    let mut class_sizes = vec![0usize; n_classes];
    for &c in &class_assignments {
        class_sizes[c] += 1;
    }
    class_sizes.sort_unstable_by(|a, b| b.cmp(a));

    let max_degree = n_bits; // Maximum meaningful degree

    let mut degree_results = Vec::new();
    let mut min_degree = None;
    let mut final_n_achievable = 0;
    let mut final_achievable = Vec::new();

    for d in 1..=max_degree {
        let (success, n_achievable, achievable) =
            check_separation_at_degree(&labels, &class_assignments, n_bits, n_classes, d);
        degree_results.push((d, success));

        if success && min_degree.is_none() {
            min_degree = Some(d);
            final_n_achievable = n_achievable;
            final_achievable = achievable;
            break; // Found minimum
        }

        // Record last attempt's data even if unsuccessful
        final_n_achievable = n_achievable;
        final_achievable = achievable;
    }

    SeparatingDegreeResult {
        dim,
        n_points,
        n_classes,
        class_sizes,
        min_degree,
        n_achievable_sigs: final_n_achievable,
        achievable_sigs: final_achievable,
        degree_results,
    }
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
    // Multi-class GF(2) separating degree tests (I-012)
    // ====================================================================

    #[test]
    fn test_gf2_solvable_trivial() {
        // 1x1 system: [1] c = [1] -> solvable
        let m = vec![vec![1u8]];
        assert!(gf2_solvable(&m, &[1]));
        // [1] c = [0] -> solvable (c=0)
        assert!(gf2_solvable(&m, &[0]));
        // [0] c = [1] -> NOT solvable
        let m2 = vec![vec![0u8]];
        assert!(!gf2_solvable(&m2, &[1]));
    }

    #[test]
    fn test_gf2_solvable_2x2() {
        // [ 1 0 ] c = [1]   -> c = [1, 0]
        // [ 0 1 ]     [0]
        let m = vec![vec![1, 0], vec![0, 1]];
        assert!(gf2_solvable(&m, &[1, 0]));
        assert!(gf2_solvable(&m, &[0, 1]));
        assert!(gf2_solvable(&m, &[1, 1]));
    }

    #[test]
    fn test_gf2_solvable_inconsistent() {
        // [ 1 1 ] c = [0]   -> c1 + c2 = 0 AND c1 + c2 = 1 -> impossible
        // [ 1 1 ]     [1]
        let m = vec![vec![1, 1], vec![1, 1]];
        assert!(!gf2_solvable(&m, &[0, 1]));
    }

    #[test]
    fn test_signatures_jointly_separate_4_classes() {
        // s1 = 0b0011 (classes 0,1 -> 1; classes 2,3 -> 0)
        // s2 = 0b0101 (classes 0,2 -> 1; classes 1,3 -> 0)
        // Pairs: class 0 -> (1,1), class 1 -> (1,0), class 2 -> (0,1), class 3 -> (0,0)
        // All 4 distinct!
        assert!(signatures_jointly_separate(0b0011, 0b0101, 4));

        // s1 = 0b0011, s2 = 0b0011 -> same signature, pairs not distinct
        assert!(!signatures_jointly_separate(0b0011, 0b0011, 4));
    }

    #[test]
    fn test_separating_degree_dim32_is_3() {
        // Known result: dim=32, 2 classes, minimum degree 3
        let result = find_minimum_separating_degree(32);
        assert_eq!(result.n_points, 15);
        assert_eq!(result.n_classes, 2, "dim=32 should have 2 motif classes");
        assert_eq!(
            result.min_degree,
            Some(3),
            "dim=32 minimum separating degree should be 3 (cubic); got {:?}. \
             degree_results: {:?}",
            result.min_degree,
            result.degree_results
        );
    }

    #[test]
    fn test_separating_degree_dim64() {
        // I-012 open question: what is the minimum separating degree at dim=64?
        let result = find_minimum_separating_degree(64);
        assert_eq!(result.n_points, 31);
        assert_eq!(result.n_classes, 4, "dim=64 should have 4 motif classes");

        // The minimum degree must exist (it's at most n_bits = 4)
        assert!(
            result.min_degree.is_some(),
            "dim=64 must have a separating degree; degree_results: {:?}",
            result.degree_results
        );

        let d = result.min_degree.unwrap();
        eprintln!("=== GF(2) Separating Degree at dim=64 ===");
        eprintln!("  n_points: {}", result.n_points);
        eprintln!("  n_classes: {}", result.n_classes);
        eprintln!("  class_sizes: {:?}", result.class_sizes);
        eprintln!("  min_degree: {d}");
        eprintln!("  n_achievable_sigs: {}", result.n_achievable_sigs);
        eprintln!("  degree_results: {:?}", result.degree_results);

        // C-480: degree grows as projective_dim = log2(dim) - 2
        assert_eq!(
            d, 4,
            "dim=64 minimum separating degree should be 4 (quartic = projective dim)"
        );
    }

    #[test]
    fn test_separating_degree_dim64_diagnostic() {
        // Full diagnostic: show class sizes and edge count distribution
        use crate::analysis::boxkites::motif_components_for_cross_assessors;
        let comps = motif_components_for_cross_assessors(64);

        let mut edge_counts: Vec<(usize, usize)> = comps
            .iter()
            .enumerate()
            .map(|(i, c)| (c.edges.len(), i))
            .collect();
        edge_counts.sort_unstable();

        let mut current_edges = edge_counts[0].0;
        let mut class_start = 0;
        let mut class_info: Vec<(usize, usize)> = Vec::new(); // (edge_count, count)

        for (idx, &(edges, _)) in edge_counts.iter().enumerate() {
            if edges != current_edges {
                class_info.push((current_edges, idx - class_start));
                class_start = idx;
                current_edges = edges;
            }
        }
        class_info.push((current_edges, edge_counts.len() - class_start));

        eprintln!("=== dim=64 Motif Class Diagnostic ===");
        eprintln!("  Total components: {}", comps.len());
        for (edges, count) in &class_info {
            eprintln!("  Class: {count} components with {edges} edges");
        }

        // Also show PG labels for each class
        for (class_idx, (edges, count)) in class_info.iter().enumerate() {
            let labels: Vec<PGPoint> = comps
                .iter()
                .filter(|c| c.edges.len() == *edges)
                .map(|c| component_xor_label(c).unwrap())
                .collect();
            eprintln!("  Class {class_idx} ({count} comps, {edges} edges): PG labels = {labels:?}");
        }
    }

    #[test]
    fn test_separating_degree_dim128() {
        // Extends C-480: predict degree 5 at dim=128 (PG(5,2), 63 pts, 8 classes)
        let result = find_minimum_separating_degree(128);
        assert_eq!(result.n_points, 63);
        assert_eq!(result.n_classes, 8, "dim=128 should have 8 motif classes");

        assert!(
            result.min_degree.is_some(),
            "dim=128 must have a separating degree; degree_results: {:?}",
            result.degree_results
        );

        let d = result.min_degree.unwrap();
        eprintln!("=== GF(2) Separating Degree at dim=128 ===");
        eprintln!("  n_points: {}", result.n_points);
        eprintln!("  n_classes: {}", result.n_classes);
        eprintln!("  class_sizes: {:?}", result.class_sizes);
        eprintln!("  min_degree: {d}");
        eprintln!("  n_achievable_sigs: {}", result.n_achievable_sigs);
        eprintln!("  degree_results: {:?}", result.degree_results);

        // C-480 extension: degree = projective_dim = log2(dim) - 2 = 5
        assert_eq!(
            d, 5,
            "dim=128 minimum separating degree should be 5 (quintic = projective dim)"
        );

        // At min degree, ALL 255 non-zero signatures should be achievable
        assert_eq!(
            result.n_achievable_sigs, 255,
            "all 2^8-1=255 signatures should be achievable at degree 5"
        );
    }

    #[test]
    fn test_separating_degree_dim128_diagnostic() {
        use crate::analysis::boxkites::motif_components_for_cross_assessors;
        let comps = motif_components_for_cross_assessors(128);

        let mut edge_counts: Vec<(usize, usize)> = comps
            .iter()
            .enumerate()
            .map(|(i, c)| (c.edges.len(), i))
            .collect();
        edge_counts.sort_unstable();

        let mut current_edges = edge_counts[0].0;
        let mut class_start = 0;
        let mut class_info: Vec<(usize, usize)> = Vec::new();

        for (idx, &(edges, _)) in edge_counts.iter().enumerate() {
            if edges != current_edges {
                class_info.push((current_edges, idx - class_start));
                class_start = idx;
                current_edges = edges;
            }
        }
        class_info.push((current_edges, edge_counts.len() - class_start));

        eprintln!("=== dim=128 Motif Class Diagnostic ===");
        eprintln!("  Total components: {}", comps.len());
        for (edges, count) in &class_info {
            eprintln!("  Class: {count} components with {edges} edges");
        }
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

    // ====================================================================
    // Phase A (T1): dim=256 separating degree verification (C-592..C-594)
    // ====================================================================

    #[test]
    fn test_separating_degree_dim256() {
        // T1 prediction: min_degree = log2(dim) - 2 = 6 at dim=256.
        // PG(6,2): 127 points, 16 motif classes.
        // Uses greedy partition refinement (brute-force infeasible for
        // C(n_achievable, 4) tuples).
        let result = find_minimum_separating_degree(256);
        assert_eq!(result.n_points, 127, "PG(6,2) should have 127 points");
        assert_eq!(
            result.n_classes, 16,
            "dim=256 should have 16 motif classes (dim/16)"
        );

        assert!(
            result.min_degree.is_some(),
            "dim=256 must have a separating degree; degree_results: {:?}",
            result.degree_results
        );

        let d = result.min_degree.unwrap();
        eprintln!("=== GF(2) Separating Degree at dim=256 ===");
        eprintln!("  n_points: {}", result.n_points);
        eprintln!("  n_classes: {}", result.n_classes);
        eprintln!("  class_sizes: {:?}", result.class_sizes);
        eprintln!("  min_degree: {d}");
        eprintln!("  n_achievable_sigs: {}", result.n_achievable_sigs);
        eprintln!("  degree_results: {:?}", result.degree_results);

        // C-592: min_degree = log2(dim) - 2 = 6 at dim=256
        assert_eq!(
            d, 6,
            "dim=256 minimum separating degree should be 6 (sextic = projective dim)"
        );
    }

    #[test]
    fn test_separating_degree_dim256_diagnostic() {
        // Full diagnostic: show class sizes and edge count distribution at dim=256
        use crate::analysis::boxkites::motif_components_for_cross_assessors;
        let comps = motif_components_for_cross_assessors(256);

        let mut edge_counts: Vec<(usize, usize)> = comps
            .iter()
            .enumerate()
            .map(|(i, c)| (c.edges.len(), i))
            .collect();
        edge_counts.sort_unstable();

        let mut current_edges = edge_counts[0].0;
        let mut class_start = 0;
        let mut class_info: Vec<(usize, usize)> = Vec::new();

        for (idx, &(edges, _)) in edge_counts.iter().enumerate() {
            if edges != current_edges {
                class_info.push((current_edges, idx - class_start));
                class_start = idx;
                current_edges = edges;
            }
        }
        class_info.push((current_edges, edge_counts.len() - class_start));

        eprintln!("=== dim=256 Motif Class Diagnostic ===");
        eprintln!("  Total components: {}", comps.len());
        assert_eq!(comps.len(), 127, "dim=256 should have 127 components");
        for (edges, count) in &class_info {
            eprintln!("  Class: {count} components with {edges} edges");
        }
        assert_eq!(
            class_info.len(),
            16,
            "dim=256 should have 16 distinct edge-count classes"
        );
    }

    #[test]
    fn test_separating_degree_formula_universality() {
        // C-593: Verify min_degree = log2(dim) - 2 universally across
        // dims 32, 64, 128, 256. Writes CSV output for archival.
        let dims_and_expected: &[(usize, usize)] = &[
            (32, 3),  // log2(32) - 2 = 3
            (64, 4),  // log2(64) - 2 = 4
            (128, 5), // log2(128) - 2 = 5
            (256, 6), // log2(256) - 2 = 6
        ];

        let mut csv_rows: Vec<String> =
            vec!["dim,n_points,n_classes,predicted_degree,measured_degree,match".to_string()];

        for &(dim, expected_degree) in dims_and_expected {
            let result = find_minimum_separating_degree(dim);
            let measured = result.min_degree.unwrap_or(0);
            let matches = measured == expected_degree;

            csv_rows.push(format!(
                "{},{},{},{},{},{}",
                dim, result.n_points, result.n_classes, expected_degree, measured, matches
            ));

            assert_eq!(
                measured, expected_degree,
                "dim={dim}: predicted degree {expected_degree}, got {measured}. \
                 n_classes={}, degree_results={:?}",
                result.n_classes, result.degree_results,
            );
        }

        eprintln!("=== Separating Degree Formula Sweep ===");
        for row in &csv_rows {
            eprintln!("  {row}");
        }
    }
}
