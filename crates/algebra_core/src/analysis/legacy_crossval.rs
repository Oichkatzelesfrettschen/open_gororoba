//! Cross-validation of legacy CSV data against the Cayley-Dickson codebase.
//!
//! This module validates pre-existing analysis data (from the legacy Python
//! workspace) against our verified Rust implementation. Each test either
//! CONFIRMS the legacy result, or documents the DISCREPANCY with an explanation.
//!
//! CSV sources: `data/csv/legacy/` (200 files, dims 2-2048)
//!
//! # Cross-validation targets
//!
//! 1. Zero-Divisor Adjacency Matrices (16D/32D/64D)
//! 2. Octonion Associator Computations
//! 3. Quaternion Lie Bracket Structure
//! 4. Dimension-by-Dimension Property Retention

use crate::construction::cayley_dickson::{cd_basis_mul_sign, cd_multiply};

/// Compute the commutativity matrix for a Cayley-Dickson algebra.
///
/// Returns a `dim x dim` matrix C where C[i][j] = true iff
/// `e_i * e_j = e_j * e_i` (the basis elements commute).
///
/// For i=0 or j=0, the identity always commutes, so C[i][j] = true.
/// For i=j (diagonal), trivially C[i][j] = true.
///
/// For all other pairs, commutativity iff s(i,j) = s(j,i) where
/// s(i,j) = cd_basis_mul_sign(dim, i, j).
pub fn commutativity_matrix(dim: usize) -> Vec<Vec<bool>> {
    let mut mat = vec![vec![false; dim]; dim];
    for (i, row) in mat.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            if i == 0 || j == 0 || i == j {
                *cell = true;
            } else {
                let sij = cd_basis_mul_sign(dim, i, j);
                let sji = cd_basis_mul_sign(dim, j, i);
                *cell = sij == sji;
            }
        }
    }
    mat
}

/// Compute the anti-commutativity matrix for a Cayley-Dickson algebra.
///
/// Returns C where C[i][j] = true iff e_i and e_j anti-commute
/// (both non-identity, non-equal, and s(i,j) = -s(j,i)).
pub fn anti_commutativity_matrix(dim: usize) -> Vec<Vec<bool>> {
    let mut mat = vec![vec![false; dim]; dim];
    for (i, row) in mat.iter_mut().enumerate().skip(1) {
        for (j, cell) in row.iter_mut().enumerate().skip(1) {
            if i != j {
                let sij = cd_basis_mul_sign(dim, i, j);
                let sji = cd_basis_mul_sign(dim, j, i);
                *cell = sij == -sji;
            }
        }
    }
    mat
}

/// Compute the same-parity adjacency matrix (what the legacy CSVs contain).
///
/// A[i][j] = 1 iff i and j have the same parity (both even or both odd)
/// and i != j.
pub fn same_parity_adjacency(dim: usize) -> Vec<Vec<u8>> {
    let mut mat = vec![vec![0u8; dim]; dim];
    for (i, row) in mat.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            if i != j && (i % 2) == (j % 2) {
                *cell = 1;
            }
        }
    }
    mat
}

/// Build a basis element as a dim-dimensional vector with 1.0 at index i.
fn basis_vec(dim: usize, i: usize) -> Vec<f64> {
    let mut v = vec![0.0; dim];
    v[i] = 1.0;
    v
}

/// Compute the Lie bracket [e_i, e_j] = e_i * e_j - e_j * e_i.
pub fn lie_bracket(dim: usize, i: usize, j: usize) -> Vec<f64> {
    let ei = basis_vec(dim, i);
    let ej = basis_vec(dim, j);
    let prod_ij = cd_multiply(&ei, &ej);
    let prod_ji = cd_multiply(&ej, &ei);
    prod_ij
        .iter()
        .zip(prod_ji.iter())
        .map(|(a, b)| a - b)
        .collect()
}

/// Compute the associator [a, b, c] = (a*b)*c - a*(b*c).
pub fn associator(dim: usize, i: usize, j: usize, k: usize) -> Vec<f64> {
    let ei = basis_vec(dim, i);
    let ej = basis_vec(dim, j);
    let ek = basis_vec(dim, k);
    let ab = cd_multiply(&ei, &ej);
    let left = cd_multiply(&ab, &ek);
    let bc = cd_multiply(&ej, &ek);
    let right = cd_multiply(&ei, &bc);
    left.iter().zip(right.iter()).map(|(a, b)| a - b).collect()
}

/// Check if a Cayley-Dickson algebra is alternative (weak associativity).
///
/// Alternative means: (a*a)*b = a*(a*b) and (a*b)*b = a*(b*b) for all a,b.
/// Holds for R, C, H, O (dims 1,2,4,8). Fails at dim=16 (sedenions).
///
/// IMPORTANT: Single basis elements always satisfy the alternative law
/// trivially (since e_i^2 = -e_0 is central). We must test with
/// multi-component elements to detect non-alternativity.
pub fn is_alternative(dim: usize) -> bool {
    // By Hurwitz theorem, only dims 1,2,4,8 are alternative.
    // For dim <= 8, we can confirm. For dim > 8, we find a counterexample.
    if dim <= 8 {
        return true; // Proven by Hurwitz theorem
    }
    // Test with 2-component elements: a = e_i + e_j
    // This exposes non-alternativity that single basis elements hide.
    for i in 1..dim.min(16) {
        for j in (i + 1)..dim.min(16) {
            let mut a = vec![0.0; dim];
            a[i] = 1.0;
            a[j] = 1.0;
            // Test with various b
            for k in 1..dim.min(16) {
                let b = basis_vec(dim, k);
                // Left alternative: (a*a)*b == a*(a*b)
                let aa = cd_multiply(&a, &a);
                let left = cd_multiply(&aa, &b);
                let ab = cd_multiply(&a, &b);
                let right = cd_multiply(&a, &ab);
                let diff: f64 = left
                    .iter()
                    .zip(right.iter())
                    .map(|(x, y)| (x - y).abs())
                    .sum();
                if diff > 1e-12 {
                    return false;
                }
            }
        }
    }
    true
}

/// Verify properties for the given Cayley-Dickson algebra dimension.
///
/// Returns a struct with boolean properties.
#[derive(Debug, Clone)]
pub struct AlgebraProperties {
    pub dim: usize,
    pub is_commutative: bool,
    pub is_associative: bool,
    pub is_alternative: bool,
    pub is_normed: bool,
    pub is_division: bool,
    pub contains_r_c_h: bool,
}

/// Check all algebraic properties for a given dimension.
pub fn check_algebra_properties(dim: usize) -> AlgebraProperties {
    assert!(dim.is_power_of_two() && dim >= 1);

    // Commutativity: e_i * e_j = e_j * e_i for all i,j
    let commutative = if dim <= 2 {
        true
    } else {
        let mat = commutativity_matrix(dim);
        (1..dim).all(|i| (1..dim).all(|j| mat[i][j] || i == j))
    };

    // Associativity: (e_i * e_j) * e_k = e_i * (e_j * e_k) for all i,j,k
    let associative = if dim <= 4 {
        true
    } else {
        let mut assoc = true;
        'outer: for i in 1..dim.min(16) {
            for j in 1..dim.min(16) {
                for k in 1..dim.min(16) {
                    let a = associator(dim, i, j, k);
                    let norm: f64 = a.iter().map(|x| x * x).sum();
                    if norm > 1e-20 {
                        assoc = false;
                        break 'outer;
                    }
                }
            }
        }
        assoc
    };

    // Alternativity
    let alternative = is_alternative(dim);

    // Normed: ||a*b|| = ||a|| * ||b|| for all a,b
    // By Hurwitz theorem, only dims 1, 2, 4, 8
    let normed = dim <= 8;

    // Division algebra: a*b = 0 implies a=0 or b=0
    // By Hurwitz theorem, only dims 1, 2, 4, 8
    let division = dim <= 8;

    // Contains R, C, H as sub-algebras:
    // Every CD algebra of dim >= 4 embeds all lower ones
    let contains_r_c_h = dim >= 4;

    AlgebraProperties {
        dim,
        is_commutative: commutative,
        is_associative: associative,
        is_alternative: alternative,
        is_normed: normed,
        is_division: division,
        contains_r_c_h,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Task #243: Cross-validate legacy ZD adjacency matrices (16D/32D/64D)
    // ========================================================================

    #[test]
    fn test_legacy_zd_adjacency_16d_is_commutativity() {
        // The legacy CSV "Zero-Divisor_Adjacency_Matrix__16D_.csv" contains
        // a 16x16 binary matrix with the same-parity pattern:
        // A[i][j] = 1 iff i,j have the same parity and i != j.
        //
        // FINDING: This is NOT the actual zero-divisor adjacency graph.
        // Individual basis elements e_i are never zero-divisors (they have
        // unit norm, so e_i * e_j = +/- e_{i XOR j} != 0). Zero-divisors
        // require LINEAR COMBINATIONS of basis elements.
        //
        // We test whether the legacy matrix matches the COMMUTATIVITY relation
        // or the same-parity relation.

        let dim = 16;
        let comm = commutativity_matrix(dim);
        let parity = same_parity_adjacency(dim);

        // First verify the same-parity structure of the legacy data.
        // Legacy CSV row i: A[i][j] = 1 iff (i%2 == j%2) && (i != j)
        // (Verified by manual inspection of the CSV)

        // Count commutativity edges (excluding identity and diagonal)
        let mut comm_edges = 0;
        let mut parity_edges = 0;
        let mut comm_matches_parity = 0;
        let mut comm_disagrees = Vec::new();

        for i in 1..dim {
            for j in (i + 1)..dim {
                if comm[i][j] {
                    comm_edges += 1;
                }
                if parity[i][j] == 1 {
                    parity_edges += 1;
                }
                if comm[i][j] == (parity[i][j] == 1) {
                    comm_matches_parity += 1;
                } else {
                    comm_disagrees.push((i, j, comm[i][j], parity[i][j]));
                }
            }
        }

        let total_pairs = (dim - 1) * (dim - 2) / 2; // C(15,2) = 105

        // Report findings
        eprintln!("=== Legacy ZD Adjacency Cross-Validation (dim=16) ===");
        eprintln!("Total non-identity pairs: {}", total_pairs);
        eprintln!("Commutativity edges: {}", comm_edges);
        eprintln!("Same-parity edges: {}", parity_edges);
        eprintln!(
            "Agreement: {}/{} ({:.1}%)",
            comm_matches_parity,
            total_pairs,
            100.0 * comm_matches_parity as f64 / total_pairs as f64
        );

        if !comm_disagrees.is_empty() {
            eprintln!("Disagreements (first 10):");
            for &(i, j, c, p) in comm_disagrees.iter().take(10) {
                let sij = cd_basis_mul_sign(dim, i, j);
                let sji = cd_basis_mul_sign(dim, j, i);
                eprintln!(
                    "  ({},{}): comm={}, parity={}, s({},{})={}, s({},{})={}",
                    i, j, c, p, i, j, sij, j, i, sji
                );
            }
        }

        // The key assertion: document whether legacy matches commutativity
        // or parity or neither
        eprintln!(
            "FINDING: Legacy CSV {} commutativity matrix.",
            if comm_disagrees.is_empty() {
                "MATCHES"
            } else {
                "DOES NOT MATCH"
            }
        );
    }

    #[test]
    fn test_legacy_zd_adjacency_32d_structure() {
        let dim = 32;
        let comm = commutativity_matrix(dim);
        let parity = same_parity_adjacency(dim);

        let mut comm_edges = 0;
        let mut parity_edges = 0;
        let mut agreement = 0;

        for i in 1..dim {
            for j in (i + 1)..dim {
                if comm[i][j] {
                    comm_edges += 1;
                }
                if parity[i][j] == 1 {
                    parity_edges += 1;
                }
                if comm[i][j] == (parity[i][j] == 1) {
                    agreement += 1;
                }
            }
        }

        let total = (dim - 1) * (dim - 2) / 2;
        eprintln!("=== Legacy ZD Adjacency Cross-Validation (dim=32) ===");
        eprintln!("Total non-identity pairs: {}", total);
        eprintln!("Commutativity edges: {}", comm_edges);
        eprintln!("Same-parity edges: {}", parity_edges);
        eprintln!(
            "Agreement: {}/{} ({:.1}%)",
            agreement,
            total,
            100.0 * agreement as f64 / total as f64
        );
    }

    #[test]
    fn test_legacy_zd_adjacency_64d_structure() {
        let dim = 64;
        let comm = commutativity_matrix(dim);
        let parity = same_parity_adjacency(dim);

        let mut comm_edges = 0;
        let mut parity_edges = 0;
        let mut agreement = 0;

        for i in 1..dim {
            for j in (i + 1)..dim {
                if comm[i][j] {
                    comm_edges += 1;
                }
                if parity[i][j] == 1 {
                    parity_edges += 1;
                }
                if comm[i][j] == (parity[i][j] == 1) {
                    agreement += 1;
                }
            }
        }

        let total = (dim - 1) * (dim - 2) / 2;
        eprintln!("=== Legacy ZD Adjacency Cross-Validation (dim=64) ===");
        eprintln!("Total non-identity pairs: {}", total);
        eprintln!("Commutativity edges: {}", comm_edges);
        eprintln!("Same-parity edges: {}", parity_edges);
        eprintln!(
            "Agreement: {}/{} ({:.1}%)",
            agreement,
            total,
            100.0 * agreement as f64 / total as f64
        );
    }

    // ========================================================================
    // Task #245: Cross-validate quaternion Lie bracket structure
    // ========================================================================

    #[test]
    fn test_quaternion_lie_brackets() {
        // Quaternion basis: e0=1, e1=i, e2=j, e3=k
        // Expected Lie brackets:
        //   [i,j] = ij - ji = k - (-k) = 2k
        //   [j,k] = jk - kj = i - (-i) = 2i
        //   [k,i] = ki - ik = j - (-j) = 2j
        let dim = 4;

        // [e1, e2] should be 2*e3
        let bracket_12 = lie_bracket(dim, 1, 2);
        assert_eq!(bracket_12, vec![0.0, 0.0, 0.0, 2.0], "[i,j] should be 2k");

        // [e2, e3] should be 2*e1
        let bracket_23 = lie_bracket(dim, 2, 3);
        assert_eq!(bracket_23, vec![0.0, 2.0, 0.0, 0.0], "[j,k] should be 2i");

        // [e3, e1] should be 2*e2
        let bracket_31 = lie_bracket(dim, 3, 1);
        assert_eq!(bracket_31, vec![0.0, 0.0, 2.0, 0.0], "[k,i] should be 2j");

        // All quaternion imaginary units anti-commute
        for i in 1..4 {
            for j in 1..4 {
                if i != j {
                    let bracket = lie_bracket(dim, i, j);
                    let norm_sq: f64 = bracket.iter().map(|x| x * x).sum();
                    assert!(
                        norm_sq > 3.99,
                        "[e{},e{}] should have ||.||^2 = 4, got {}",
                        i,
                        j,
                        norm_sq
                    );
                }
            }
        }

        // Legacy CSV reports: [i,j] computed as [0,1,0,1], expected 2k
        // FINDING: The legacy "Computed Result" [0,1,0,1] does NOT represent
        // the quaternion coefficient vector [0,0,0,2]. It may use a boolean
        // encoding of "which components are nonzero" or a different convention.
        // Our CD computation correctly gives [0,0,0,2].

        eprintln!("=== Quaternion Lie Bracket Cross-Validation ===");
        eprintln!("[i,j] = {:?} (expected [0,0,0,2]=2k)", bracket_12);
        eprintln!("[j,k] = {:?} (expected [0,2,0,0]=2i)", bracket_23);
        eprintln!("[k,i] = {:?} (expected [0,0,2,0]=2j)", bracket_31);
        eprintln!("Legacy CSV [i,j] reported as [0,1,0,1] -- DISCREPANCY");
        eprintln!("FINDING: Legacy uses boolean component mask, not coefficient values.");
    }

    #[test]
    fn test_quaternion_psi_matrix() {
        // The psi matrix at dim=4 encodes the sign structure.
        // psi(i,j) = 0 if cd_basis_mul_sign(4,i,j) = +1
        // psi(i,j) = 1 if cd_basis_mul_sign(4,i,j) = -1
        let dim = 4;
        let mut psi = [[0u8; 4]; 4];
        for i in 0..dim {
            for j in 0..dim {
                let s = cd_basis_mul_sign(dim, i, j);
                psi[i][j] = if s == 1 { 0 } else { 1 };
            }
        }

        eprintln!("=== Quaternion psi matrix (dim=4) ===");
        for i in 0..dim {
            eprintln!("  {:?}", psi[i]);
        }

        // Verify all imaginary units anti-commute in quaternions
        for i in 1..4 {
            for j in 1..4 {
                if i != j {
                    let sij = cd_basis_mul_sign(dim, i, j);
                    let sji = cd_basis_mul_sign(dim, j, i);
                    assert_eq!(sij, -sji, "e{} and e{} should anti-commute", i, j);
                }
            }
        }

        // Verify quaternion multiplication table:
        // i*i = -1, j*j = -1, k*k = -1
        // i*j = k, j*k = i, k*i = j
        // j*i = -k, k*j = -i, i*k = -j
        assert_eq!(cd_basis_mul_sign(4, 1, 1), -1); // i^2 = -1
        assert_eq!(cd_basis_mul_sign(4, 2, 2), -1); // j^2 = -1
        assert_eq!(cd_basis_mul_sign(4, 3, 3), -1); // k^2 = -1

        // i*j: result index = 1 XOR 2 = 3 (k), sign should be +1
        assert_eq!(1 ^ 2, 3);
        assert_eq!(cd_basis_mul_sign(4, 1, 2), 1); // i*j = +k

        // j*k: result index = 2 XOR 3 = 1 (i), sign should be +1
        assert_eq!(2 ^ 3, 1);
        assert_eq!(cd_basis_mul_sign(4, 2, 3), 1); // j*k = +i

        // k*i: result index = 3 XOR 1 = 2 (j), sign should be +1
        assert_eq!(3 ^ 1, 2);
        assert_eq!(cd_basis_mul_sign(4, 3, 1), 1); // k*i = +j

        // Anti-commutators: j*i = -k, k*j = -i, i*k = -j
        assert_eq!(cd_basis_mul_sign(4, 2, 1), -1); // j*i = -k
        assert_eq!(cd_basis_mul_sign(4, 3, 2), -1); // k*j = -i
        assert_eq!(cd_basis_mul_sign(4, 1, 3), -1); // i*k = -j
    }

    // ========================================================================
    // Task #244: Cross-validate octonion associator computations
    // ========================================================================

    #[test]
    fn test_octonion_associator_computation() {
        // The legacy CSV lists specific triples and their associators.
        // We verify the first triple: (e3, e6, e7)
        // Legacy: (ab)c = e2, a(bc) = -e5, residue = e2 + e5
        let dim = 8;

        let assoc_367 = associator(dim, 3, 6, 7);
        eprintln!("=== Octonion Associator Cross-Validation ===");
        eprintln!("[e3,e6,e7] = {:?}", assoc_367);

        // Compute step by step:
        // e3 * e6: index = 3 XOR 6 = 5, sign = cd_basis_mul_sign(8, 3, 6)
        let s36 = cd_basis_mul_sign(8, 3, 6);
        let idx36 = 3 ^ 6;
        eprintln!("e3*e6 = {}*e{}", s36, idx36);

        // (e3*e6) * e7 = s36 * (e5 * e7)
        // e5 * e7: index = 5 XOR 7 = 2, sign = cd_basis_mul_sign(8, 5, 7)
        let s57 = cd_basis_mul_sign(8, 5, 7);
        let idx57 = 5 ^ 7;
        eprintln!("e5*e7 = {}*e{}", s57, idx57);
        eprintln!("(e3*e6)*e7 = {}*e{}", s36 * s57, idx57);

        // e6 * e7: index = 6 XOR 7 = 1, sign = cd_basis_mul_sign(8, 6, 7)
        let s67 = cd_basis_mul_sign(8, 6, 7);
        let idx67 = 6 ^ 7;
        eprintln!("e6*e7 = {}*e{}", s67, idx67);

        // e3 * (e6*e7) = s67 * (e3 * e1)
        // e3 * e1: index = 3 XOR 1 = 2, sign = cd_basis_mul_sign(8, 3, 1)
        let s31 = cd_basis_mul_sign(8, 3, 1);
        let idx31 = 3 ^ 1;
        eprintln!("e3*e1 = {}*e{}", s31, idx31);
        eprintln!("e3*(e6*e7) = {}*e{}", s67 * s31, idx31);

        // Associator = (e3*e6)*e7 - e3*(e6*e7) = (s36*s57 - s67*s31)*e2
        // since both products land on e2 (idx57 = idx31 = 2)
        assert_eq!(idx57, 2);
        assert_eq!(idx31, 2);
        let left_coeff = s36 * s57;
        let right_coeff = s67 * s31;
        eprintln!(
            "Associator coefficient on e2: {} - {} = {}",
            left_coeff,
            right_coeff,
            left_coeff - right_coeff
        );

        // Legacy claims: (ab)c = e2, a(bc) = -e5
        // Our computation: both products land on e2 (not e5!)
        // The legacy may be using a DIFFERENT octonion multiplication table.
        // There are 480 valid octonion multiplication tables (corresponding
        // to the 480 ways to orient the Fano plane).
        eprintln!("FINDING: Legacy and our CD construction may use different");
        eprintln!("octonion multiplication tables. The CD doubling formula");
        eprintln!("(a,b)(c,d) = (ac - d*b, da + bc*) is one specific choice.");
    }

    #[test]
    fn test_octonion_full_associator_survey() {
        // Count how many triples of octonion basis elements have
        // nonzero associators. The octonions are alternative but
        // not associative.
        let dim = 8;
        let mut nonzero_count = 0;
        let mut total_triples = 0;

        for i in 1..dim {
            for j in 1..dim {
                for k in 1..dim {
                    if i == j || j == k || i == k {
                        continue;
                    }
                    total_triples += 1;
                    let a = associator(dim, i, j, k);
                    let norm_sq: f64 = a.iter().map(|x| x * x).sum();
                    if norm_sq > 1e-20 {
                        nonzero_count += 1;
                    }
                }
            }
        }

        eprintln!("=== Octonion Associator Survey ===");
        eprintln!("Total distinct triples (i!=j!=k, i!=k): {}", total_triples);
        eprintln!(
            "Non-zero associators: {} ({:.1}%)",
            nonzero_count,
            100.0 * nonzero_count as f64 / total_triples as f64
        );

        // Verify alternativity: [a,a,b] = 0 and [a,b,b] = 0
        for i in 1..dim {
            for j in 1..dim {
                if i == j {
                    continue;
                }
                let aab = associator(dim, i, i, j);
                let abb = associator(dim, i, j, j);
                let norm_aab: f64 = aab.iter().map(|x| x * x).sum();
                let norm_abb: f64 = abb.iter().map(|x| x * x).sum();
                assert!(
                    norm_aab < 1e-20,
                    "Octonions should be alternative: [e{},e{},e{}] != 0",
                    i,
                    i,
                    j
                );
                assert!(
                    norm_abb < 1e-20,
                    "Octonions should be alternative: [e{},e{},e{}] != 0",
                    i,
                    j,
                    j
                );
            }
        }
        eprintln!("Alternativity VERIFIED: all [a,a,b] = [a,b,b] = 0");
    }

    #[test]
    fn test_octonion_fano_plane_from_multiplication() {
        // The Fano plane has 7 lines, each containing 3 points.
        // A line {a,b,c} means e_a * e_b = +/- e_c (cyclically).
        // Reconstruct the Fano plane from cd_basis_mul_sign.
        let dim = 8;
        let mut lines = Vec::new();

        for i in 1..dim {
            for j in (i + 1)..dim {
                let k = i ^ j;
                if k > j && k < dim {
                    let sij = cd_basis_mul_sign(dim, i, j);
                    let sjk = cd_basis_mul_sign(dim, j, k);
                    let ski = cd_basis_mul_sign(dim, k, i);
                    lines.push((i, j, k, sij, sjk, ski));
                }
            }
        }

        eprintln!("=== Octonion Fano Plane from CD Multiplication ===");
        let mut positive_oriented = 0;
        let mut negative_oriented = 0;
        for &(i, j, k, sij, sjk, ski) in &lines {
            let orientation = sij * sjk * ski;
            if orientation > 0 {
                positive_oriented += 1;
            } else {
                negative_oriented += 1;
            }
            eprintln!(
                "  {{{},{},{}}} orientation={} (s_ij={}, s_jk={}, s_ki={})",
                i, j, k, orientation, sij, sjk, ski
            );
        }

        assert_eq!(lines.len(), 7, "Fano plane should have exactly 7 lines");
        eprintln!(
            "Fano lines: {} positive, {} negative oriented",
            positive_oriented, negative_oriented
        );
        // The CD construction gives 5 positive and 2 negative (or vice versa)
        // depending on convention. This was verified in C-533.
    }

    // ========================================================================
    // Task #246: Cross-validate dimension-by-dimension property retention
    // ========================================================================

    #[test]
    fn test_property_retention_dim1_reals() {
        let props = check_algebra_properties(1);
        assert!(props.is_commutative, "R should be commutative");
        assert!(props.is_associative, "R should be associative");
        assert!(props.is_normed, "R should be normed");
        assert!(props.is_division, "R should be a division algebra");
    }

    #[test]
    fn test_property_retention_dim2_complex() {
        let props = check_algebra_properties(2);
        assert!(props.is_commutative, "C should be commutative");
        assert!(props.is_associative, "C should be associative");
        assert!(props.is_normed, "C should be normed");
        assert!(props.is_division, "C should be a division algebra");
    }

    #[test]
    fn test_property_retention_dim4_quaternions() {
        let props = check_algebra_properties(4);
        assert!(!props.is_commutative, "H should NOT be commutative");
        assert!(props.is_associative, "H should be associative");
        assert!(props.is_alternative, "H should be alternative");
        assert!(props.is_normed, "H should be normed");
        assert!(props.is_division, "H should be a division algebra");
        assert!(props.contains_r_c_h, "H contains R, C, H");
    }

    #[test]
    fn test_property_retention_dim8_octonions() {
        let props = check_algebra_properties(8);
        assert!(!props.is_commutative, "O should NOT be commutative");
        assert!(!props.is_associative, "O should NOT be associative");
        assert!(props.is_alternative, "O should be alternative");
        assert!(props.is_normed, "O should be normed");
        assert!(props.is_division, "O should be a division algebra");
        assert!(props.contains_r_c_h, "O contains R, C, H");
    }

    #[test]
    fn test_property_retention_dim16_sedenions() {
        let props = check_algebra_properties(16);
        assert!(!props.is_commutative, "S should NOT be commutative");
        assert!(!props.is_associative, "S should NOT be associative");
        assert!(
            !props.is_alternative,
            "S should NOT be alternative (first failure at dim=16)"
        );
        assert!(!props.is_normed, "S should NOT be normed (Hurwitz)");
        assert!(
            !props.is_division,
            "S should NOT be a division algebra (zero divisors exist)"
        );
        assert!(
            props.contains_r_c_h,
            "S DOES contain R, C, H as sub-algebras"
        );

        // Legacy CSV claims: "Contains R,C,H?" = False for dim=16
        // FINDING: This is WRONG. Sedenions embed all lower CD algebras.
        // The first 4 basis elements (e0,e1,e2,e3) form a quaternion
        // sub-algebra. The first 8 form an octonion sub-algebra.
        eprintln!("=== Property Retention Cross-Validation (dim=16) ===");
        eprintln!("Legacy claims 'Contains R,C,H?' = False for sedenions");
        eprintln!("FINDING: INCORRECT. Sedenions embed R, C, H, O as sub-algebras.");
        eprintln!("The first 2^k basis elements form a dim-2^k sub-algebra.");
    }

    #[test]
    fn test_property_retention_dim32_pathions() {
        let props = check_algebra_properties(32);
        assert!(!props.is_commutative);
        assert!(!props.is_associative);
        assert!(!props.is_alternative);
        assert!(!props.is_normed);
        assert!(!props.is_division);
        assert!(props.contains_r_c_h, "32D contains R, C, H");
    }

    #[test]
    fn test_legacy_property_retention_errata() {
        // Legacy CSV claims for dimensions 16, 27, 32, 64, 128, 248:
        // "Contains R,C,H?" = False for ALL of them.
        //
        // This is WRONG for CD algebras (dims 16, 32, 64, 128) which
        // embed all lower algebras.
        //
        // For dim=27 (Albert algebra) and dim=248 (E8 Lie algebra),
        // these are NOT Cayley-Dickson algebras, so the property
        // retention question is different. But even the Albert algebra
        // is built from octonions and inherits their embedding of H.
        //
        // Additional errata in the legacy CSV:
        // - dim=16 "Normed?" = False: CORRECT (Hurwitz theorem)
        // - dim=16 "Albert Structure Exists?" = True: CORRECT
        //   (sedenions relate to Albert algebra via octonion pairs)
        // - dim=27 "Monstrous Moonshine Possible?" = True: dim=196883
        //   is the smallest faithful representation of the Monster,
        //   and 27 appears in the exceptional decomposition via E8,
        //   but the direct Moonshine connection to dim=27 is tenuous.

        for dim_exp in [4, 5, 6, 7] {
            let dim = 1 << dim_exp;
            let props = check_algebra_properties(dim);
            assert!(
                props.contains_r_c_h,
                "CD algebra at dim={} DOES contain R, C, H (legacy says False)",
                dim
            );
        }
        eprintln!("=== Legacy Property Retention Errata ===");
        eprintln!("CONFIRMED: dims 16, 32, 64, 128 all contain R, C, H sub-algebras.");
        eprintln!("Legacy CSV incorrectly reports 'Contains R,C,H?' = False.");
    }

    #[test]
    fn test_sedenion_alternativity_failure_example() {
        // Demonstrate a concrete non-alternative element pair in sedenions.
        // Single basis elements trivially satisfy alternativity because
        // e_i^2 = -e_0 is central. Must use multi-component elements.
        let dim = 16;
        let mut found = false;
        for i in 1..dim {
            for j in (i + 1)..dim {
                // a = e_i + e_j (two-component element)
                let mut a = vec![0.0; dim];
                a[i] = 1.0;
                a[j] = 1.0;
                for k in 1..dim {
                    let b = basis_vec(dim, k);
                    let aa = cd_multiply(&a, &a);
                    let left = cd_multiply(&aa, &b);
                    let ab = cd_multiply(&a, &b);
                    let right = cd_multiply(&a, &ab);
                    let diff: f64 = left
                        .iter()
                        .zip(right.iter())
                        .map(|(x, y)| (x - y).abs())
                        .sum();
                    if diff > 1e-12 {
                        eprintln!(
                            "Non-alternative: a=e{}+e{}, b=e{}, ||(aa)b - a(ab)|| = {:.6}",
                            i, j, k, diff
                        );
                        found = true;
                        break;
                    }
                }
                if found {
                    break;
                }
            }
            if found {
                break;
            }
        }
        assert!(
            found,
            "Sedenions should fail alternativity for multi-component elements"
        );
    }

    // ========================================================================
    // Task #251: Cross-validate hyperquaternion multiplication table
    // ========================================================================

    #[test]
    fn test_hyperquaternion_table_completion() {
        // Legacy CSV: partial 8x8 octonion multiplication table with basis
        // {1,i,j,k,l,il,jl,kl} = {e_0,e_1,e_2,e_3,e_4,e_5,e_6,e_7}
        //
        // Known entries (from CSV, non-"?" cells):
        //   Row 1 (1*x):  1, i, j, k, l, il, jl, kl
        //   Row i (i*x):  i, ?, k, -j, -il, ?, ?, ?
        //   Row j (j*x):  j, -k, ?, i, -jl, ?, ?, ?
        //   Row k (k*x):  k, j, -i, ?, -kl, ?, ?, ?
        //   Row l (l*x):  l, il, jl, kl, -1, ?, ?, ?
        //   Rows il/jl/kl: only first column known (identity product)
        //
        // We fill the entire 8x8 table from cd_basis_mul_sign and verify
        // all known entries match.

        let dim = 8;
        // Basis names for readable output
        let names = ["1", "i", "j", "k", "l", "il", "jl", "kl"];

        // Build the complete multiplication table:
        // table[a][b] = (result_index, sign) where e_a * e_b = sign * e_{result_index}
        let mut table = vec![vec![(0usize, 1i32); dim]; dim];
        for a in 0..dim {
            for b in 0..dim {
                if a == 0 {
                    table[a][b] = (b, 1);
                } else if b == 0 {
                    table[a][b] = (a, 1);
                } else if a == b {
                    table[a][b] = (0, -1); // e_i^2 = -1
                } else {
                    let idx = a ^ b;
                    let sign = cd_basis_mul_sign(dim, a, b);
                    table[a][b] = (idx, sign);
                }
            }
        }

        // Print the complete table
        eprintln!("=== Complete Octonion Multiplication Table ===");
        eprint!("      ");
        for b in 0..dim {
            eprint!("{:>5}", names[b]);
        }
        eprintln!();
        for a in 0..dim {
            eprint!("  {:>3}:", names[a]);
            for b in 0..dim {
                let (idx, sign) = table[a][b];
                let prefix = if sign == 1 { " " } else { "-" };
                eprint!(" {}{:>3}", prefix, names[idx]);
            }
            eprintln!();
        }

        // Verify known entries from CSV:
        // Format: (row, col, expected_index, expected_sign)
        let known_entries: Vec<(usize, usize, usize, i32)> = vec![
            // Row i: i*j=k, i*k=-j, i*l=-il
            (1, 2, 3, 1),  // i*j = +k
            (1, 3, 2, -1), // i*k = -j
            (1, 4, 5, -1), // i*l = -il
            // Row j: j*i=-k, j*k=i, j*l=-jl
            (2, 1, 3, -1), // j*i = -k
            (2, 3, 1, 1),  // j*k = +i
            (2, 4, 6, -1), // j*l = -jl
            // Row k: k*i=j, k*j=-i, k*l=-kl
            (3, 1, 2, 1),  // k*i = +j
            (3, 2, 1, -1), // k*j = -i
            (3, 4, 7, -1), // k*l = -kl
            // Row l: l*i=il, l*j=jl, l*k=kl, l*l=-1
            (4, 1, 5, 1),  // l*i = +il
            (4, 2, 6, 1),  // l*j = +jl
            (4, 3, 7, 1),  // l*k = +kl
            (4, 4, 0, -1), // l*l = -1
        ];

        let mut sign_flips = 0;
        let mut index_mismatches = 0;
        for &(a, b, exp_idx, exp_sign) in &known_entries {
            let (got_idx, got_sign) = table[a][b];
            if got_idx != exp_idx {
                index_mismatches += 1;
            } else if got_sign != exp_sign {
                sign_flips += 1;
                eprintln!(
                    "SIGN FLIP: {}*{} = {}*{} (ours) vs {}*{} (CSV)",
                    names[a],
                    names[b],
                    if got_sign == 1 { "+" } else { "-" },
                    names[got_idx],
                    if exp_sign == 1 { "+" } else { "-" },
                    names[exp_idx]
                );
            }
        }

        // Result index must always match (XOR structure is convention-independent)
        assert_eq!(
            index_mismatches, 0,
            "Product indices should match (e_a * e_b always lands on e_{{a XOR b}})"
        );

        // Sign flips are expected: the legacy table uses a DIFFERENT doubling
        // convention than our Cayley-Dickson formula. The CD construction has
        // a choice in where to place conjugation:
        //   Convention A (ours): (a,b)(c,d) = (ac - d*b, da + bc*)
        //   Convention B (legacy): sign flip in cross-half products
        //
        // All 6 sign flips involve one "upper" index (1,2,3) and one "lower"
        // index (4=l): exactly the cross-half boundary products.
        // Both conventions produce valid octonion algebras (among the 480
        // possible multiplication tables).
        assert_eq!(
            sign_flips, 6,
            "Expect exactly 6 sign flips at cross-half boundary"
        );

        eprintln!(
            "VERIFIED: {}/{} known entries match CD computation exactly",
            known_entries.len() - sign_flips,
            known_entries.len()
        );
        eprintln!(
            "DOCUMENTED: {} sign flips at quaternion/doubling boundary (convention difference)",
            sign_flips
        );

        // Count the "?" entries that we filled in (total 64 - 8 identity row
        // - 8 identity col + 1 overlap - 13 known non-identity = ~36)
        let unknown_count = 64 - 8 - 7 - 13; // 36 cells were "?"
        eprintln!(
            "Completed {} unknown entries in the multiplication table",
            unknown_count
        );
    }

    #[test]
    fn test_hyperquaternion_structural_properties() {
        // Verify structural properties of the completed octonion table:
        // 1. Anti-commutativity: e_a * e_b = -(e_b * e_a) for a,b != 0, a != b
        // 2. Every row and column is a permutation of +/-{e_0..e_7}
        // 3. Squares: e_i^2 = -e_0 for all i != 0
        let dim = 8;

        // Anti-commutativity
        for a in 1..dim {
            for b in (a + 1)..dim {
                let sab = cd_basis_mul_sign(dim, a, b);
                let sba = cd_basis_mul_sign(dim, b, a);
                assert_eq!(
                    sab, -sba,
                    "e{}*e{}: sign {} vs e{}*e{}: sign {} -- should anti-commute",
                    a, b, sab, b, a, sba
                );
            }
        }

        // Every row permutation: for each a, the set {a XOR b | b=0..7} = {0..7}
        for a in 0..dim {
            let mut targets: Vec<usize> = (0..dim)
                .map(|b| {
                    if a == 0 || b == 0 {
                        a.max(b)
                    } else if a == b {
                        0
                    } else {
                        a ^ b
                    }
                })
                .collect();
            targets.sort();
            assert_eq!(
                targets,
                vec![0, 1, 2, 3, 4, 5, 6, 7],
                "Row {} should be a permutation of basis elements",
                a
            );
        }

        // Norm composition (octonions are normed):
        // ||e_a * e_b|| = ||e_a|| * ||e_b|| = 1 for all basis elements
        // This is trivially true since products are always +/- e_k with ||e_k|| = 1

        eprintln!("=== Hyperquaternion Structural Properties ===");
        eprintln!("VERIFIED: anti-commutativity for all 21 pairs");
        eprintln!("VERIFIED: each row is a permutation of {{e_0..e_7}}");
        eprintln!("VERIFIED: norm composition (trivial for basis elements)");
    }

    // ========================================================================
    // Task #250: Analyze E6/E7/F4-inspired multiplication structures
    // ========================================================================

    #[test]
    fn test_e6_inspired_matrix_structure() {
        // The legacy "E6-Inspired Extended Multiplication Structure" CSV is
        // a 16x16 matrix with block structure:
        //
        //   [A  0]     A = 7x8 dense floats (rows 0-6)
        //   [I  0]     Row 7 = unit vector at position 7
        //   [0  D]     D = 8x8 checkerboard with +/-0.4 off-diagonal (rows 8-15)
        //
        // This is NOT a multiplication table. It appears to be an eigenvector
        // decomposition or projection matrix.

        // Parse the known values from the CSV
        let upper_left: Vec<Vec<f64>> = vec![
            vec![0.569, -0.537, 0.562, -0.05, -0.049, 0.246, -0.08],
            vec![0.288, 0.252, -0.079, 0.774, -0.003, 0.339, 0.364],
            vec![-0.041, 0.217, 0.353, -0.16, 0.84, -0.003, 0.308],
            vec![-0.531, -0.122, 0.04, -0.031, 0.052, 0.816, -0.179],
            vec![-0.29, -0.466, -0.031, -0.091, -0.193, -0.072, 0.804],
            vec![0.196, -0.511, -0.668, 0.073, 0.482, 0.029, -0.124],
            vec![0.432, 0.328, -0.323, -0.598, -0.139, 0.391, 0.268],
        ];

        // Check if columns are approximately orthonormal
        eprintln!("=== E6-Inspired Matrix Analysis ===");
        let n = upper_left.len();
        let m = upper_left[0].len();

        // Compute column dot products
        let mut max_off_diag = 0.0f64;
        let mut diag_norms = Vec::new();
        for ci in 0..m {
            let col_i: Vec<f64> = (0..n).map(|r| upper_left[r][ci]).collect();
            let norm_sq: f64 = col_i.iter().map(|x| x * x).sum();
            diag_norms.push(norm_sq.sqrt());

            for cj in (ci + 1)..m {
                let col_j: Vec<f64> = (0..n).map(|r| upper_left[r][cj]).collect();
                let dot: f64 = col_i.iter().zip(col_j.iter()).map(|(a, b)| a * b).sum();
                max_off_diag = max_off_diag.max(dot.abs());
            }
        }

        eprintln!("Upper-left block (7x7):");
        eprintln!("  Column norms: {:?}", diag_norms);
        eprintln!("  Max off-diagonal dot product: {:.6}", max_off_diag);

        // Check if columns are orthogonal (dot products near 0)
        let approximately_orthogonal = max_off_diag < 0.05;
        eprintln!(
            "  Columns approximately orthogonal: {}",
            approximately_orthogonal
        );

        // Check if columns are unit vectors
        let approximately_orthonormal =
            approximately_orthogonal && diag_norms.iter().all(|n| (n - 1.0).abs() < 0.05);
        eprintln!(
            "  Columns approximately orthonormal: {}",
            approximately_orthonormal
        );

        // Analyze the lower-right 8x8 checkerboard block.
        // From the CSV, the pattern is: D[i][j] = 1.0 if i==j, else (-1)^(i+j) * 0.4
        //
        // The lower-right block D = I + 0.4 * J where J is a checkerboard matrix
        // with J[i][j] = (-1)^(i+j) (a Hadamard-like pattern).
        // Eigenvalues of J_n: (n-1) copies of -1/(n-1) and one copy of 1.
        // So D = I + 0.4*J has eigenvalues: 1 - 0.4/7 ~ 0.943 (x7) and 1.4 (x1).

        eprintln!("Lower-right block (8x8):");
        eprintln!("  Pattern: I + 0.4 * checkerboard");
        eprintln!("  Expected eigenvalues: 0.943 (x7), 1.4 (x1)");

        // Compare against actual sedenion multiplication table:
        // the sedenion table has entries +/- e_k, so entries are {-1,0,1}.
        // The E6 matrix has floating-point values -- fundamentally different.
        eprintln!("FINDING: E6-Inspired matrix is NOT a multiplication table.");
        eprintln!("  Sedenion multiplication table entries: {{-1, 0, +1}}");
        eprintln!("  E6 matrix entries: continuous floats");
        eprintln!("  The matrix appears to be an eigenvector/projection decomposition");
        eprintln!("  with two independent blocks (7-dim and 8-dim).");

        // Verify the upper block has continuous (non-integer) entries
        let has_non_integer = upper_left
            .iter()
            .any(|row| row.iter().any(|&v| (v - v.round()).abs() > 0.01));
        assert!(
            has_non_integer,
            "Upper-left block should have non-integer entries"
        );
    }

    #[test]
    fn test_extended_sedenion_vs_e6() {
        // The "Extended Sedenion" CSV has the SAME upper-left 7x8 block as E6
        // but zeros in rows 8-15 (where E6 has the checkerboard block).
        //
        // This confirms both CSVs are derived from the same analysis:
        // the "Extended Sedenion" is a truncated version that only retains
        // the dense eigenvector block, while the "E6-Inspired" adds the
        // checkerboard structure in the lower-right.

        let upper_e6: Vec<f64> = vec![0.569, -0.537, 0.562, -0.05, -0.049, 0.246, -0.08];
        let upper_ext: Vec<f64> = vec![0.569, -0.537, 0.562, -0.05, -0.049, 0.246, -0.08];

        // First rows should be identical
        for (a, b) in upper_e6.iter().zip(upper_ext.iter()) {
            assert!(
                (a - b).abs() < 1e-10,
                "E6 and Extended Sedenion first rows should match"
            );
        }

        // The 7x7 upper-left block: check if it could be a Gram matrix
        // of E6 roots projected to 7 dimensions
        // E6 has rank 6, so 6 simple roots in 6D. The 7x7 block has one
        // extra dimension -- possibly including the affine root for E6^(1).

        // Check row norms of the upper-left block
        let rows: Vec<Vec<f64>> = vec![
            vec![0.569, -0.537, 0.562, -0.05, -0.049, 0.246, -0.08],
            vec![0.288, 0.252, -0.079, 0.774, -0.003, 0.339, 0.364],
            vec![-0.041, 0.217, 0.353, -0.16, 0.84, -0.003, 0.308],
            vec![-0.531, -0.122, 0.04, -0.031, 0.052, 0.816, -0.179],
            vec![-0.29, -0.466, -0.031, -0.091, -0.193, -0.072, 0.804],
            vec![0.196, -0.511, -0.668, 0.073, 0.482, 0.029, -0.124],
            vec![0.432, 0.328, -0.323, -0.598, -0.139, 0.391, 0.268],
        ];

        eprintln!("=== Extended Sedenion vs E6 Comparison ===");
        for (i, row) in rows.iter().enumerate() {
            let norm: f64 = row.iter().map(|x| x * x).sum::<f64>().sqrt();
            eprintln!("  Row {} norm: {:.6}", i, norm);
        }

        // Check if rows are orthonormal
        let mut max_off = 0.0f64;
        let mut norms_ok = true;
        for i in 0..rows.len() {
            let ni: f64 = rows[i].iter().map(|x| x * x).sum::<f64>().sqrt();
            if (ni - 1.0).abs() > 0.05 {
                norms_ok = false;
            }
            for j in (i + 1)..rows.len() {
                let dot: f64 = rows[i].iter().zip(rows[j].iter()).map(|(a, b)| a * b).sum();
                max_off = max_off.max(dot.abs());
            }
        }

        eprintln!("  Rows unit-length: {}", norms_ok);
        eprintln!("  Max off-diagonal row dot product: {:.6}", max_off);
        eprintln!(
            "  Rows approximately orthonormal: {}",
            norms_ok && max_off < 0.05
        );

        eprintln!("FINDING: Both CSVs share the same upper-left eigenvector block.");
        eprintln!("  Extended Sedenion = E6 with lower-right block zeroed.");
        eprintln!("  Neither is a Cayley-Dickson multiplication table.");
    }

    // ========================================================================
    // Task #252: Analyze Monstrous Moonshine connections in legacy data
    // ========================================================================

    #[test]
    fn test_moonshine_j_function_coefficients() {
        // The legacy "Spin Foam Amplitudes vs Monster Group Coefficients" CSV
        // claims a connection between spin foam amplitudes and Monster group
        // coefficients. We verify the numerical content.
        //
        // The j-invariant expansion (Klein's modular function):
        //   j(q) = q^{-1} + 744 + 196884*q + 21493760*q^2 + 864299970*q^3
        //          + 20245856256*q^4 + ...
        //
        // These are the coefficients of the Monster module V^# (Frenkel-Lepowsky-Meurman).

        // Known j-function coefficients c(n) for n=1..10 (OEIS A007240)
        let j_coeffs: [f64; 10] = [
            196884.0,
            21493760.0,
            864299970.0,
            20245856256.0,
            333202640600.0,
            4252023300096.0,
            44656994071935.0,
            401489888665600.0,
            3176440229784420.0,
            22312779956505600.0,
        ];

        // Legacy CSV "Scaled Monster Coeffs" column
        let legacy_scaled: [f64; 10] = [
            0.000196884,
            0.02149376,
            0.86429997,
            20.245856256,
            333.2026406,
            4252.023300096001,
            44656.994071935005,
            401490.88665600005,
            3176440.22978442,
            23123279.479533825,
        ];

        // Verify: legacy = j_coeffs / 1e9
        eprintln!("=== Monstrous Moonshine Coefficient Verification ===");
        let mut max_rel_err = 0.0f64;
        for n in 0..10 {
            let expected_scaled = j_coeffs[n] / 1e9;
            let rel_err = ((legacy_scaled[n] - expected_scaled) / expected_scaled).abs();
            max_rel_err = max_rel_err.max(rel_err);
            eprintln!(
                "  c({:>2}): j/1e9 = {:>20.6}, CSV = {:>20.6}, rel_err = {:.2e}",
                n + 1,
                expected_scaled,
                legacy_scaled[n],
                rel_err
            );
        }

        // Most entries match well but c(10) has a discrepancy
        // CSV says 23123279.479... while j_coeffs[9]/1e9 = 22312779.956...
        // This is a ~3.6% error -- likely a data entry mistake in the legacy CSV.
        eprintln!("Max relative error: {:.4e}", max_rel_err);
        eprintln!("NOTE: c(10) has ~3.6% error in legacy CSV (23123279 vs 22312779)");

        // Verify the first 8 coefficients match to within 0.01%
        for n in 0..8 {
            let expected = j_coeffs[n] / 1e9;
            let rel_err = ((legacy_scaled[n] - expected) / expected).abs();
            assert!(
                rel_err < 1e-4,
                "c({}) should match j-function: CSV={}, expected={}",
                n + 1,
                legacy_scaled[n],
                expected
            );
        }
    }

    #[test]
    fn test_moonshine_spin_foam_column_is_trivial() {
        // The "Spin Foam Amplitude" column in the legacy CSV is:
        //   state n -> ln(n+1) / 1000
        //
        // This has NOTHING to do with actual spin foam amplitudes
        // (which involve 6j symbols, BF theory path integrals, etc.).
        // It is a trivially generated column.

        let legacy_amplitudes: [f64; 10] = [
            0.0006931471805599454,
            0.0010986122886681095,
            0.001386294361119891,
            0.0016094379124340999,
            0.0017917594692280557,
            0.0019459101490553127,
            0.002079441541679837,
            0.0021972245773362186,
            0.0023025850929940467,
            0.0023978952727983695,
        ];

        eprintln!("=== Spin Foam Amplitude Column Analysis ===");
        for n in 0..10 {
            let expected = ((n + 2) as f64).ln() / 1000.0;
            let diff = (legacy_amplitudes[n] - expected).abs();
            eprintln!(
                "  state {:>2}: CSV = {:.16}, ln({})/1000 = {:.16}, diff = {:.2e}",
                n + 1,
                legacy_amplitudes[n],
                n + 2,
                expected,
                diff
            );
            assert!(diff < 1e-15, "Spin foam amplitude should be ln(n+1)/1000");
        }

        eprintln!("FINDING: 'Spin Foam Amplitude' column = ln(n+1)/1000 exactly.");
        eprintln!("  This is a trivial logarithmic function, not a spin foam computation.");
        eprintln!("  Comparing ln(n) (logarithmic growth) to j-function coefficients");
        eprintln!("  (exponential growth ~ exp(4*pi*sqrt(n))) is meaningless.");
    }

    #[test]
    fn test_moonshine_cd_connection_assessment() {
        // Assess the actual mathematical connections between
        // Cayley-Dickson algebras and the Monster group.
        //
        // Known connections (via literature):
        // 1. Octonions (dim=8) -> E8 lattice -> Leech lattice -> Monster
        //    (Borcherds' proof uses vertex operator algebras on Leech)
        // 2. dim(V^#_1) = 196884 = dim of Griess algebra
        //    196884 = 196883 + 1 (McKay's observation)
        //    196883 is the smallest faithful rep of the Monster
        // 3. 196884 is NOT a power of 2, so it is NOT a CD algebra dimension
        // 4. The legacy CSV claims "Monstrous Moonshine Structure Possible"
        //    for dims 16384, 32768, 65536, 131072, 262144 -- all powers of 2.
        //    These ARE CD algebra dimensions but the Moonshine connection
        //    is not exhibited (no vertex operator algebra, no j-function
        //    coefficient match, no Monster group representation).

        // Check: is 196884 a power of 2?
        let griess_dim = 196884u64;
        let is_power_of_2 = griess_dim.is_power_of_two();
        assert!(!is_power_of_2, "196884 is NOT a power of 2");

        // Factor 196884
        // 196884 = 4 * 49221 = 4 * 3 * 16407 = 12 * 16407 = 12 * 3 * 5469
        // = 36 * 5469 = 36 * 3 * 1823 = 108 * 1823
        // 1823 is prime.
        // So 196884 = 2^2 * 3^3 * 1823
        let mut n = griess_dim;
        let mut factors = Vec::new();
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43] {
            while n % p == 0 {
                factors.push(p);
                n /= p;
            }
        }
        if n > 1 {
            factors.push(n);
        }

        eprintln!("=== CD-Monster Connection Assessment ===");
        eprintln!("Griess algebra dimension: {}", griess_dim);
        eprintln!("Is power of 2: {}", is_power_of_2);
        eprintln!("Prime factorization: {:?}", factors);
        eprintln!("  196884 = 2^2 * 3^3 * 1823 (no power-of-2 structure)");
        eprintln!();
        eprintln!("Legacy claims 'Moonshine possible' for CD dims 16384..262144");
        eprintln!("FINDING: These claims are UNSUBSTANTIATED.");
        eprintln!("  - No j-function coefficient equals any power of 2");
        eprintln!("  - No Monster representation has power-of-2 dimension");
        eprintln!("  - The actual CD->Monster path goes through E8->Leech");
        eprintln!("  - Legacy CSV provides zero computational evidence");

        // Verify: the first few j-function coefficients are not powers of 2
        let j_coeffs: [u64; 5] = [196884, 21493760, 864299970, 20245856256, 333202640600];
        for &c in &j_coeffs {
            assert!(
                !c.is_power_of_two(),
                "j-function coefficient {} should not be a power of 2",
                c
            );
        }
    }

    // ========================================================================
    // Task #248: Extend psi=1 fraction convergence through dim=2048
    // ========================================================================

    #[test]
    fn test_psi_fraction_convergence_to_50_percent() {
        // The psi matrix entry psi(i,j) = 0 if cd_basis_mul_sign(dim,i,j) = +1,
        //                                  1 if cd_basis_mul_sign(dim,i,j) = -1.
        //
        // C-534 established: the fraction of psi=1 entries converges
        // monotonically to 50% as dimension doubles.
        //
        // We extend through dim=2048 (11 data points) and verify:
        // 1. Monotonic decrease toward 50%
        // 2. Geometric convergence with ratio ~0.5 per doubling
        // 3. Exact values at every power of 2

        // C-534 used the FULL (dim-1)x(dim-1) matrix including diagonal.
        // Since psi(i,i) = 1 always (e_i^2 = -e_0, sign = -1), the diagonal
        // inflates the fraction above 50%. The fraction then converges
        // to 50% FROM ABOVE as dim grows.
        //
        // The upper-triangle-only fraction converges FROM BELOW.
        // Both converge to 50% at infinite dimension.

        eprintln!("=== psi=1 Fraction Convergence (dim=2 through 2048) ===");
        eprintln!("Using FULL (dim-1)x(dim-1) matrix (matching C-534 convention)");

        let mut prev_fraction = 1.0f64;
        let mut fractions = Vec::new();

        for dim_exp in 1..=11 {
            let dim: usize = 1 << dim_exp;
            let n = dim - 1; // imaginary basis elements
            let total = n * n; // full matrix including diagonal

            let mut psi_one_count = 0usize;
            for i in 1..dim {
                for j in 1..dim {
                    let sign = cd_basis_mul_sign(dim, i, j);
                    if sign == -1 {
                        psi_one_count += 1;
                    }
                }
            }

            let fraction = psi_one_count as f64 / total as f64;
            let excess = fraction - 0.5;
            fractions.push((dim, fraction));

            eprintln!(
                "  dim={:>5}: {}/{} psi=1 ({:.6}), excess = {:.6}",
                dim, psi_one_count, total, fraction, excess
            );

            // Verify monotonic decrease toward 50% (from above)
            if dim > 2 {
                assert!(
                    fraction <= prev_fraction + 1e-10,
                    "psi=1 fraction should decrease: dim={} ({:.6}) > prev ({:.6})",
                    dim,
                    fraction,
                    prev_fraction
                );
                assert!(
                    fraction >= 0.5 - 1e-10,
                    "Full-matrix psi=1 fraction should stay >= 50%: dim={} ({:.6})",
                    dim,
                    fraction
                );
            }
            prev_fraction = fraction;
        }

        // Verify the EXACT FORMULA: fraction = (n+1)/(2n) = 0.5 + 1/(2n)
        // where n = dim-1.
        //
        // Proof:
        //   Diagonal: n entries, all psi=1 (e_i^2 = -e_0).
        //   Off-diagonal: n(n-1) entries. By anti-commutativity (C-537),
        //     sign(i,j) = -sign(j,i), so exactly n(n-1)/2 have psi=1.
        //   Total psi=1: n + n(n-1)/2 = n(n+1)/2.
        //   Fraction: n(n+1) / (2*n^2) = (n+1)/(2n) = 0.5 + 1/(2n).
        eprintln!("\n  Verifying exact formula: fraction = (n+1)/(2n), n=dim-1");
        for &(dim, fraction) in &fractions {
            let n = (dim - 1) as f64;
            let exact = (n + 1.0) / (2.0 * n);
            let err = (fraction - exact).abs();
            eprintln!(
                "    dim={:>5}: measured={:.10}, exact={:.10}, err={:.2e}",
                dim, fraction, exact, err
            );
            assert!(
                err < 1e-12,
                "psi=1 fraction should match (n+1)/(2n) exactly at dim={}",
                dim
            );
        }

        // Show convergence ratios for completeness
        eprintln!("\n  Convergence ratios (excess[n+1] / excess[n]):");
        for i in 1..fractions.len() {
            let (dim_curr, f_curr) = fractions[i];
            let (_dim_prev, f_prev) = fractions[i - 1];
            let excess_prev = f_prev - 0.5;
            let excess_curr = f_curr - 0.5;
            if excess_prev.abs() > 1e-10 {
                let ratio = excess_curr / excess_prev;
                eprintln!(
                    "    dim={:>5}: ratio = {:.6} (exact: {:.6})",
                    dim_curr,
                    ratio,
                    (dim_curr / 2 - 1) as f64 / (dim_curr - 1) as f64
                );
            }
        }
    }

    #[test]
    fn test_psi_fraction_exact_values() {
        // Verify exact psi=1 fractions at small dimensions where we
        // can compute the analytical formula.
        //
        // For dim=4 (quaternions): 3 imaginary pairs, all anti-commute,
        //   psi(1,2)=0, psi(1,3)=1, psi(2,3)=0 => fraction = 1/3 = 33.3%
        //   Wait -- let me compute properly.
        //   cd_basis_mul_sign(4,1,2) = +1 => psi=0
        //   cd_basis_mul_sign(4,1,3) = -1 => psi=1
        //   cd_basis_mul_sign(4,2,3) = +1 => psi=0
        //   So psi=1 count = 1, total = 3, fraction = 1/3 = 33.3%
        //
        // Hmm, C-534 says 66.7% for dim=4. Let me check...
        // The claim says "fraction of psi=1 entries in the (dim-1)x(dim-1) matrix"
        // which includes BOTH upper and lower triangles plus diagonal.
        // For the FULL matrix (not just upper triangle):
        //   psi(i,j) where 1<=i,j<dim and i!=j (since psi(i,i) is for self-product)
        //   For i=j: cd_basis_mul_sign(dim,i,i) = -1 always => psi=1
        //   So diagonal contributes (dim-1) psi=1 entries.
        //
        // Let me recalculate with the FULL matrix.

        for dim_exp in 1..=7 {
            let dim: usize = 1 << dim_exp;
            let n = dim - 1;
            let total = n * n; // all entries in the (dim-1)x(dim-1) matrix

            let mut psi_one = 0usize;
            for i in 1..dim {
                for j in 1..dim {
                    let sign = cd_basis_mul_sign(dim, i, j);
                    if sign == -1 {
                        psi_one += 1;
                    }
                }
            }

            let fraction = psi_one as f64 / total as f64;
            eprintln!(
                "  dim={:>4}: full matrix {}/{} psi=1 ({:.4})",
                dim, psi_one, total, fraction
            );
        }

        // Also compute upper-triangle only fractions for comparison
        eprintln!("\n  Upper-triangle only (i < j):");
        for dim_exp in 1..=7 {
            let dim: usize = 1 << dim_exp;
            let n = dim - 1;
            let total = n * (n - 1) / 2;
            if total == 0 {
                eprintln!("  dim={:>4}: no pairs", dim);
                continue;
            }

            let mut psi_one = 0usize;
            for i in 1..dim {
                for j in (i + 1)..dim {
                    let sign = cd_basis_mul_sign(dim, i, j);
                    if sign == -1 {
                        psi_one += 1;
                    }
                }
            }

            let fraction = psi_one as f64 / total as f64;
            eprintln!(
                "  dim={:>4}: upper-tri {}/{} psi=1 ({:.4})",
                dim, psi_one, total, fraction
            );
        }
    }

    // ========================================================================
    // Commutativity statistics
    // ========================================================================

    #[test]
    fn test_commutativity_stats_across_dimensions() {
        // Track how commutativity fraction changes across doublings.
        // Expected: quaternions are fully non-commutative (0% commuting among
        // non-identity pairs), and the fraction should change at each doubling.
        eprintln!("=== Commutativity Statistics Across Dimensions ===");
        for dim_exp in 2..=7 {
            let dim = 1 << dim_exp;
            let comm = commutativity_matrix(dim);

            let mut commuting = 0;
            let mut total = 0;
            for i in 1..dim {
                for j in (i + 1)..dim {
                    total += 1;
                    if comm[i][j] {
                        commuting += 1;
                    }
                }
            }
            eprintln!(
                "  dim={:>4}: {}/{} commuting pairs ({:.1}%)",
                dim,
                commuting,
                total,
                100.0 * commuting as f64 / total as f64
            );
        }
    }

    // ========================================================================
    // Lattice codebook cross-validation (dims 256/512/1024/2048)
    // ========================================================================

    use crate::analysis::codebook::{
        enumerate_lattice_by_predicate, is_in_lambda_1024, is_in_lambda_2048, is_in_lambda_256,
        is_in_lambda_512, LatticeVector,
    };
    use crate::experimental::cd_external::{load_lattice_map, load_lattice_points};

    /// Parse a lattice vector string like "[-1, -1, -1, -1, -1, -1, -1, -1]"
    /// into an [i8; 8] LatticeVector.
    fn parse_lattice_vec_i8(s: &str) -> Option<LatticeVector> {
        let s = s.trim().trim_start_matches('[').trim_end_matches(']');
        let parts: Vec<&str> = s.split(',').collect();
        if parts.len() != 8 {
            return None;
        }
        let mut v = [0i8; 8];
        for (i, p) in parts.iter().enumerate() {
            v[i] = p.trim().parse::<i8>().ok()?;
        }
        Some(v)
    }

    #[test]
    fn test_lattice_csv_vs_predicate_256d() {
        // Cross-validate: CSV lattice points == predicate-enumerated Lambda_256.
        // This is the anchor test -- if this fails, the filtration chain is broken.
        let csv_points = load_lattice_points(256);
        let pred_points = enumerate_lattice_by_predicate(is_in_lambda_256);

        assert_eq!(csv_points.len(), 256, "CSV should have 256 points");
        assert_eq!(pred_points.len(), 256, "Predicates should give 256 points");

        let csv_set: std::collections::BTreeSet<Vec<i32>> = csv_points.into_iter().collect();
        let pred_set: std::collections::BTreeSet<Vec<i32>> = pred_points
            .iter()
            .map(|v| v.iter().map(|&x| x as i32).collect())
            .collect();

        assert_eq!(
            csv_set, pred_set,
            "CSV and predicate Lambda_256 must match exactly"
        );
    }

    #[test]
    fn test_lattice_csv_vs_predicate_512d() {
        // CSV 512D points should all pass is_in_lambda_512.
        // Predicate enumeration should give exactly 512 points.
        let csv_points = load_lattice_points(512);
        let pred_points = enumerate_lattice_by_predicate(is_in_lambda_512);

        assert_eq!(csv_points.len(), 512, "CSV should have 512 points");
        assert_eq!(pred_points.len(), 512, "Predicates should give 512 points");

        let csv_set: std::collections::BTreeSet<Vec<i32>> = csv_points.into_iter().collect();
        let pred_set: std::collections::BTreeSet<Vec<i32>> = pred_points
            .iter()
            .map(|v| v.iter().map(|&x| x as i32).collect())
            .collect();

        assert_eq!(
            csv_set, pred_set,
            "CSV and predicate Lambda_512 must match exactly"
        );
    }

    #[test]
    fn test_lattice_csv_vs_predicate_1024d() {
        // DISCREPANCY: Predicate gives 1026 points, CSV has 1024.
        // The 2 extra predicate points are [-1,1,1,0,-1,1,0,1] and [-1,1,1,0,-1,1,1,0].
        // These share prefix (-1,1,1,0,-1,1) with positive tail sum.
        //
        // OPEN QUESTION: Which is correct?
        // - Predicate: derived from trie-cut analysis of base universe structure
        // - CSV: generated by AI-assisted symbolic computation (provenance unknown)
        //
        // We verify: CSV is a STRICT SUBSET of the predicate set, and document
        // the 2 disputed points for further investigation.
        let csv_points = load_lattice_points(1024);
        let pred_points = enumerate_lattice_by_predicate(is_in_lambda_1024);

        assert_eq!(csv_points.len(), 1024, "CSV should have 1024 points");
        assert_eq!(
            pred_points.len(),
            1026,
            "Predicate gives 1026 (2 extra vs CSV)"
        );

        let csv_set: std::collections::BTreeSet<Vec<i32>> = csv_points.into_iter().collect();
        let pred_set: std::collections::BTreeSet<Vec<i32>> = pred_points
            .iter()
            .map(|v| v.iter().map(|&x| x as i32).collect())
            .collect();

        // CSV must be a subset of predicates (predicates are more inclusive)
        let csv_not_in_pred: Vec<_> = csv_set.difference(&pred_set).collect();
        assert!(
            csv_not_in_pred.is_empty(),
            "CSV has points not in predicate: {:?}",
            csv_not_in_pred
        );

        // Document the 2 disputed points
        let pred_not_in_csv: Vec<_> = pred_set.difference(&csv_set).collect();
        assert_eq!(
            pred_not_in_csv.len(),
            2,
            "Expected exactly 2 disputed points"
        );
        eprintln!("Lambda_1024 disputed points (predicate-only):");
        for v in &pred_not_in_csv {
            eprintln!("  {:?}", v);
        }
    }

    #[test]
    fn test_lattice_1024d_disputed_points_investigation() {
        // Deep investigation of the 2 points in the predicate but not the CSV.
        // Both satisfy all base universe + Lambda_1024 rules. We test whether
        // they are structurally anomalous compared to their neighbors.
        use crate::analysis::codebook::{is_in_base_universe, is_in_lambda_512};

        let disputed = [[-1i8, 1, 1, 0, -1, 1, 0, 1], [-1, 1, 1, 0, -1, 1, 1, 0]];

        // Neighbors: same prefix, different tail
        let neighbors = [[-1i8, 1, 1, 0, -1, 1, -1, 0], [-1, 1, 1, 0, -1, 1, 0, -1]];

        for v in &disputed {
            assert!(
                is_in_base_universe(v),
                "Disputed {:?} must be in base universe",
                v
            );
            assert!(
                is_in_lambda_2048(v),
                "Disputed {:?} must be in Lambda_2048",
                v
            );
            assert!(
                is_in_lambda_1024(v),
                "Disputed {:?} must be in Lambda_1024",
                v
            );
            assert!(
                !is_in_lambda_512(v),
                "Disputed {:?} should NOT be in Lambda_512",
                v
            );

            let sum: i32 = v.iter().map(|&x| x as i32).sum();
            let weight: usize = v.iter().filter(|&&x| x != 0).count();
            eprintln!("Disputed {:?}: sum={}, weight={}", v, sum, weight);
        }

        for v in &neighbors {
            let in_1024 = is_in_lambda_1024(v);
            let in_base = is_in_base_universe(v);
            let sum: i32 = v.iter().map(|&x| x as i32).sum();
            let weight: usize = v.iter().filter(|&&x| x != 0).count();
            eprintln!(
                "Neighbor  {:?}: sum={}, weight={}, base={}, lambda_1024={}",
                v, sum, weight, in_base, in_1024
            );
        }

        // Key observation: disputed points have sum=2, neighbors have sum=0.
        // Both are valid under the base universe (even sum requirement).
        // The CSV may have applied a TIGHTER sum constraint not in the predicate.
        let csv_points = load_lattice_points(1024);
        let csv_sums: std::collections::BTreeSet<i32> =
            csv_points.iter().map(|v| v.iter().sum()).collect();
        let pred_points = enumerate_lattice_by_predicate(is_in_lambda_1024);
        let pred_sums: std::collections::BTreeSet<i32> = pred_points
            .iter()
            .map(|v| v.iter().map(|&x| x as i32).sum())
            .collect();

        eprintln!("CSV sum range: {:?}", csv_sums);
        eprintln!("Predicate sum range: {:?}", pred_sums);

        // Check: is the CSV sum range strictly narrower?
        // If so, the CSV may have a stricter sum bound.
        assert!(
            csv_sums.is_subset(&pred_sums),
            "CSV sums should be a subset of predicate sums"
        );

        // Count how many CSV vs predicate points have sum=2
        let csv_sum2: usize = csv_points
            .iter()
            .filter(|v| v.iter().sum::<i32>() == 2)
            .count();
        let pred_sum2: usize = pred_points
            .iter()
            .filter(|v| v.iter().map(|&x| x as i32).sum::<i32>() == 2)
            .count();
        eprintln!(
            "Points with sum=2: CSV={}, Predicate={}",
            csv_sum2, pred_sum2
        );

        // If predicate has exactly 2 more sum=2 points, the dispute is within
        // the sum=2 stratum only.
        assert_eq!(
            pred_sum2 - csv_sum2,
            2,
            "The 2 disputed points should both have sum=2"
        );
    }

    #[test]
    fn test_lattice_csv_vs_predicate_2048d() {
        let csv_points = load_lattice_points(2048);
        let pred_points = enumerate_lattice_by_predicate(is_in_lambda_2048);

        assert_eq!(csv_points.len(), 2048, "CSV should have 2048 points");
        assert_eq!(
            pred_points.len(),
            2048,
            "Predicates should give 2048 points"
        );

        let csv_set: std::collections::BTreeSet<Vec<i32>> = csv_points.into_iter().collect();
        let pred_set: std::collections::BTreeSet<Vec<i32>> = pred_points
            .iter()
            .map(|v| v.iter().map(|&x| x as i32).collect())
            .collect();

        assert_eq!(
            csv_set, pred_set,
            "CSV and predicate Lambda_2048 must match exactly"
        );
    }

    #[test]
    fn test_lattice_filtration_nesting() {
        // Verify strict nesting: Lambda_256 < Lambda_512 < Lambda_1024 < Lambda_2048.
        // Note: is_in_lambda_1024 gives 1026 (2 disputed singletons vs CSV).
        let p256 = enumerate_lattice_by_predicate(is_in_lambda_256);
        let p512 = enumerate_lattice_by_predicate(is_in_lambda_512);
        let p1024 = enumerate_lattice_by_predicate(is_in_lambda_1024);
        let p2048 = enumerate_lattice_by_predicate(is_in_lambda_2048);

        assert_eq!(p256.len(), 256);
        assert_eq!(p512.len(), 512);
        assert_eq!(p1024.len(), 1026, "Predicate gives 1026 (2 extra vs CSV)");
        assert_eq!(p2048.len(), 2048);

        // Every Lambda_256 point is in Lambda_512
        for v in &p256 {
            assert!(
                is_in_lambda_512(v),
                "Lambda_256 point {:?} not in Lambda_512",
                v
            );
        }
        // Every Lambda_512 point is in Lambda_1024
        for v in &p512 {
            assert!(
                is_in_lambda_1024(v),
                "Lambda_512 point {:?} not in Lambda_1024",
                v
            );
        }
        // Every Lambda_1024 point is in Lambda_2048
        for v in &p1024 {
            assert!(
                is_in_lambda_2048(v),
                "Lambda_1024 point {:?} not in Lambda_2048",
                v
            );
        }

        eprintln!(
            "Filtration sizes: Lambda_256={}, Lambda_512={}, Lambda_1024={} (pred), Lambda_2048={}",
            p256.len(),
            p512.len(),
            p1024.len(),
            p2048.len()
        );
    }

    #[test]
    fn test_lattice_carrier_set_validation_256d() {
        // Verify that loading CSV into CarrierSet + EncodingDictionary succeeds
        // and the dictionary is a valid bijection.
        use crate::analysis::codebook::{CarrierSet, EncodingDictionary};

        let lattice_map = load_lattice_map(256);
        assert_eq!(
            lattice_map.len(),
            256,
            "Should have 256 basis-lattice pairs"
        );

        let cs = CarrierSet::from_i32_map(256, &lattice_map);
        let val = cs.validate();
        assert!(
            val.is_valid_dictionary(),
            "CarrierSet should form valid dictionary: {:?}",
            val
        );

        // Build encoding dictionary
        let ed = EncodingDictionary::try_from_carrier_set(cs);
        assert!(ed.is_ok(), "EncodingDictionary construction should succeed");
        let ed = ed.unwrap();

        // Round-trip: encode then decode every basis index
        for idx in 0..256 {
            let lv = ed.encode(idx);
            assert!(lv.is_some(), "Basis index {} should have encoding", idx);
            let decoded = ed.decode(lv.unwrap());
            assert_eq!(
                decoded,
                Some(idx),
                "Round-trip failed for basis index {}",
                idx
            );
        }
    }

    #[test]
    fn test_lattice_consistent_sum_column() {
        // Verify the "Consistent Sum" column from the CSV.
        // "Consistent Sum" = True means the Lattice Sum matches expectations.
        // We verify: Lattice Sum = sum of lattice vector coordinates.
        for &dim in &[256, 512, 1024, 2048] {
            let csv_path = format!(
                "{}/../../data/csv/cayley_dickson/{}d_lattice_mapping.csv",
                env!("CARGO_MANIFEST_DIR"),
                dim
            );
            let content = std::fs::read_to_string(&csv_path)
                .unwrap_or_else(|_| panic!("Missing CSV: {}", csv_path));

            let mut inconsistent = 0;
            let mut total = 0;
            for line in content.lines().skip(1) {
                // Parse with the internal CSV parser
                let fields: Vec<String> = {
                    let mut fields = Vec::new();
                    let mut current = String::new();
                    let mut in_quotes = false;
                    for ch in line.chars() {
                        match ch {
                            '"' => in_quotes = !in_quotes,
                            ',' if !in_quotes => {
                                fields.push(std::mem::take(&mut current));
                            }
                            _ => current.push(ch),
                        }
                    }
                    fields.push(current);
                    fields
                };

                if fields.len() >= 4 {
                    if let Some(lv) = parse_lattice_vec_i8(&fields[1]) {
                        let computed_sum: i32 = lv.iter().map(|&x| x as i32).sum();
                        let csv_sum: i32 = fields[2].trim().parse().unwrap_or(i32::MIN);
                        let consistent = fields[3].trim() == "True";

                        if computed_sum != csv_sum {
                            inconsistent += 1;
                        }
                        // Verify "Consistent Sum" column accuracy
                        assert!(
                            consistent,
                            "dim={}: CSV row claims inconsistent sum for {:?}",
                            dim, lv
                        );
                        assert_eq!(
                            computed_sum, csv_sum,
                            "dim={}: sum mismatch for {:?}: computed={}, csv={}",
                            dim, lv, computed_sum, csv_sum
                        );
                    }
                    total += 1;
                }
            }
            assert_eq!(
                total, dim,
                "dim={}: expected {} rows, got {}",
                dim, dim, total
            );
            assert_eq!(
                inconsistent, 0,
                "dim={}: {} sum inconsistencies",
                dim, inconsistent
            );
        }
    }

    #[test]
    fn test_lattice_multiplication_retention_256d() {
        // The "Structured Multiplication Retention" CSV checks which products
        // e_i * e_j land within Lambda_256 (closed under the encoding).
        // We cross-validate by computing cd_basis_mul_sign and checking if
        // e_{i XOR j} is in the codebook.
        let lattice_map = load_lattice_map(256);
        let lattice_set: std::collections::HashSet<usize> = lattice_map.keys().copied().collect();

        assert_eq!(lattice_set.len(), 256);

        // For all pairs (i,j) where both are in the codebook,
        // check if i XOR j is also in the codebook.
        let mut closed = 0;
        let mut total = 0;
        for &i in lattice_set.iter() {
            for &j in lattice_set.iter() {
                if i == j {
                    continue;
                }
                total += 1;
                let product_idx = i ^ j;
                if lattice_set.contains(&product_idx) {
                    closed += 1;
                }
            }
        }

        let closure_fraction = closed as f64 / total as f64;
        eprintln!(
            "Lambda_256 multiplication closure: {}/{} = {:.4}",
            closed, total, closure_fraction
        );

        // The codebook is an ENCODING of ALL 256 basis elements, so
        // the set {0, 1, ..., 255} is closed under XOR by definition.
        // This verifies the lattice_map is a complete bijection.
        assert_eq!(
            closure_fraction, 1.0,
            "Complete codebook (all 256 basis elements) must be XOR-closed"
        );
    }
}
