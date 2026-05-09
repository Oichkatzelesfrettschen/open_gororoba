//! Chevalley-Tits basis for E7 structure constants (Lyndon basis).
//!
//! Provides Jacobi-identity-consistent structure constants for E7 using the
//! Tits extraspecial sign convention. The existing `e7_structure.rs` uses
//! lexicographic ordering for signs, which is deterministic and antisymmetric
//! but does NOT generally satisfy the Jacobi identity. This module corrects
//! that via the mathematically rigorous Chevalley-Tits construction.
//!
//! For E7 (simply-laced, ADE type), the Chevalley basis structure constants
//! N(alpha, beta) are:
//! - 0 if alpha + beta is not a root
//! - +/-1 if alpha + beta is a root (|N| = p+1, with p=0 for simply-laced)
//! - Antisymmetric: N(alpha, beta) = -N(beta, alpha)
//! - Jacobi-consistent: the cyclic sum vanishes for all root triples
//!
//! The sign is determined by Tits' extraspecial 2-cocycle epsilon on the root
//! lattice, encoded as a bilinear form f: Q x Q -> Z/2Z.
//!
//! References:
//! - Carter, "Lie Algebras of Finite and Affine Type" Ch. 4.2
//! - Tits, "Sur les constantes de structure" (1966)

use std::collections::{HashMap, HashSet};

use super::{
    e7::geometry::{E7Root, generate_e7_roots},
    e8::root_system::E8Root,
};

/// Discretize coordinates to integer keys for hash-based lookup.
/// Multiplying by 2 converts half-integers to integers.
fn coord_key(coords: &[f64; 8]) -> [i64; 8] {
    let mut key = [0i64; 8];
    for (i, &c) in coords.iter().enumerate() {
        key[i] = (c * 2.0).round() as i64;
    }
    key
}

/// Check if a root is positive (first nonzero coordinate > 0 in lex order).
fn is_positive(coords: &[f64; 8]) -> bool {
    for &c in coords {
        if c > 1e-12 {
            return true;
        }
        if c < -1e-12 {
            return false;
        }
    }
    false
}

/// Lexicographic comparison of coordinate arrays with epsilon tolerance.
fn lex_cmp_f64(a: &[f64; 8], b: &[f64; 8]) -> std::cmp::Ordering {
    for i in 0..8 {
        let diff = a[i] - b[i];
        if diff > 1e-12 {
            return std::cmp::Ordering::Greater;
        }
        if diff < -1e-12 {
            return std::cmp::Ordering::Less;
        }
    }
    std::cmp::Ordering::Equal
}

/// Chevalley-Tits basis for E7 with Jacobi-consistent structure constants.
///
/// Constructed by:
/// 1. Identifying the 7 simple roots of E7 within the E8 orthogonal complement
/// 2. Computing the Cartan matrix and its inverse
/// 3. Expressing every root in the simple root basis
/// 4. Building Tits' extraspecial bilinear form on simple roots
/// 5. Evaluating epsilon(alpha, beta) = (-1)^{f(alpha, beta)}
pub struct ChevalleyBasis {
    roots: Vec<E7Root>,
    simple_root_indices: [usize; 7],
    tits_form: [[u8; 7]; 7],
    root_decomposition: HashMap<[i64; 8], [i32; 7]>,
    root_lookup: HashMap<[i64; 8], usize>,
}

impl ChevalleyBasis {
    /// Construct the Chevalley-Tits basis for E7.
    pub fn new() -> Self {
        let roots = generate_e7_roots();
        let root_lookup: HashMap<[i64; 8], usize> = roots
            .iter()
            .enumerate()
            .map(|(i, r)| (coord_key(&r.root.coords), i))
            .collect();

        let positive_indices: Vec<usize> = roots
            .iter()
            .enumerate()
            .filter(|(_, r)| is_positive(&r.root.coords))
            .map(|(i, _)| i)
            .collect();
        assert_eq!(positive_indices.len(), 63, "E7 has 63 positive roots");

        let simple_root_indices = find_simple_roots(&roots, &positive_indices, &root_lookup);
        let cartan = compute_cartan_matrix(&roots, &simple_root_indices);
        let cartan_inv = invert_7x7(cartan);
        let tits_form = build_tits_form(&cartan);
        let root_decomposition = decompose_all_roots(&roots, &simple_root_indices, &cartan_inv);

        Self {
            roots,
            simple_root_indices,
            tits_form,
            root_decomposition,
            root_lookup,
        }
    }

    /// All 126 E7 roots.
    pub fn roots(&self) -> &[E7Root] {
        &self.roots
    }

    /// The 7 simple root indices.
    pub fn simple_root_indices(&self) -> &[usize; 7] {
        &self.simple_root_indices
    }

    /// Jacobi-consistent structure constant N(alpha, beta).
    ///
    /// Returns +1.0, -1.0, or 0.0 following the Chevalley-Tits sign convention.
    pub fn structure_constant(&self, alpha: &E7Root, beta: &E7Root) -> f64 {
        let sum_coords = add_coords(&alpha.root.coords, &beta.root.coords);
        let sum_key = coord_key(&sum_coords);
        if !self.root_lookup.contains_key(&sum_key) {
            return 0.0;
        }
        let sum_norm_sq: f64 = sum_coords.iter().map(|x| x * x).sum();
        if (sum_norm_sq - 2.0).abs() > 1e-10 {
            return 0.0;
        }

        let alpha_key = coord_key(&alpha.root.coords);
        let beta_key = coord_key(&beta.root.coords);
        let n = match self.root_decomposition.get(&alpha_key) {
            Some(d) => d,
            None => return 0.0,
        };
        let m = match self.root_decomposition.get(&beta_key) {
            Some(d) => d,
            None => return 0.0,
        };

        // f(alpha, beta) = sum_{i,j} n_i * m_j * F[i][j] mod 2
        let f_val: i32 = n
            .iter()
            .enumerate()
            .flat_map(|(i, &ni)| {
                m.iter()
                    .enumerate()
                    .map(move |(j, &mj)| ni * mj * i32::from(self.tits_form[i][j]))
            })
            .sum();

        if f_val.rem_euclid(2) == 0 { 1.0 } else { -1.0 }
    }

    /// Exhaustive Jacobi identity verification over all root triples.
    ///
    /// Checks N(a,b)*N(a+b,c) + N(b,c)*N(b+c,a) + N(c,a)*N(c+a,b) = 0.
    pub fn verify_jacobi_identity(&self) -> bool {
        let n = self.roots.len();
        for i in 0..n {
            for j in (i + 1)..n {
                for k in (j + 1)..n {
                    if !self.check_jacobi_triple(i, j, k) {
                        return false;
                    }
                }
            }
        }
        true
    }

    fn check_jacobi_triple(&self, i: usize, j: usize, k: usize) -> bool {
        let a = &self.roots[i];
        let b = &self.roots[j];
        let c = &self.roots[k];

        // Skip triples where any pair sums to zero (one is the negative of another).
        // When [e_alpha, e_{-alpha}] = h_alpha (Cartan element), the Jacobi identity
        // involves the Cartan subalgebra and cannot be expressed purely as
        // N(a,b)*N(a+b,c) + cyclic = 0. Those cases require the additional term
        // -(gamma, alpha_check) * e_gamma from [e_gamma, h_alpha].
        let ab = add_coords(&a.root.coords, &b.root.coords);
        let bc = add_coords(&b.root.coords, &c.root.coords);
        let ca = add_coords(&c.root.coords, &a.root.coords);

        let is_zero = |v: &[f64; 8]| v.iter().all(|x| x.abs() < 1e-12);
        if is_zero(&ab) || is_zero(&bc) || is_zero(&ca) {
            return true;
        }

        let nab = self.structure_constant(a, b);
        let nbc = self.structure_constant(b, c);
        let nca = self.structure_constant(c, a);

        // Short-circuit: if no pairwise sums are roots, Jacobi holds trivially
        if nab == 0.0 && nbc == 0.0 && nca == 0.0 {
            return true;
        }

        let ab_root = E7Root {
            root: E8Root::new(ab),
        };
        let bc_root = E7Root {
            root: E8Root::new(bc),
        };
        let ca_root = E7Root {
            root: E8Root::new(ca),
        };

        let term1 = nab * self.structure_constant(&ab_root, c);
        let term2 = nbc * self.structure_constant(&bc_root, a);
        let term3 = nca * self.structure_constant(&ca_root, b);

        (term1 + term2 + term3).abs() < 1e-10
    }
}

impl Default for ChevalleyBasis {
    fn default() -> Self {
        Self::new()
    }
}

fn add_coords(a: &[f64; 8], b: &[f64; 8]) -> [f64; 8] {
    let mut r = [0.0; 8];
    for (d, r_d) in r.iter_mut().enumerate() {
        *r_d = a[d] + b[d];
    }
    r
}

/// Find the 7 simple roots: positive roots that are not sums of two positive roots.
fn find_simple_roots(
    roots: &[E7Root],
    positive_indices: &[usize],
    _root_lookup: &HashMap<[i64; 8], usize>,
) -> [usize; 7] {
    let positive_keys: HashSet<[i64; 8]> = positive_indices
        .iter()
        .map(|&i| coord_key(&roots[i].root.coords))
        .collect();

    let mut simple = Vec::new();
    for &idx in positive_indices {
        let r = &roots[idx];
        let mut is_decomposable = false;
        for &j in positive_indices {
            if j == idx {
                continue;
            }
            let a = &roots[j];
            let mut diff = [0.0; 8];
            for (d, diff_d) in diff.iter_mut().enumerate() {
                *diff_d = r.root.coords[d] - a.root.coords[d];
            }
            let diff_key = coord_key(&diff);
            if positive_keys.contains(&diff_key) {
                let r_key = coord_key(&r.root.coords);
                let a_key = coord_key(&a.root.coords);
                if diff_key != r_key && diff_key != a_key {
                    is_decomposable = true;
                    break;
                }
            }
        }
        if !is_decomposable {
            simple.push(idx);
        }
    }

    assert_eq!(
        simple.len(),
        7,
        "E7 should have exactly 7 simple roots, found {}",
        simple.len()
    );
    simple.sort_by(|&a, &b| lex_cmp_f64(&roots[a].root.coords, &roots[b].root.coords));
    [
        simple[0], simple[1], simple[2], simple[3], simple[4], simple[5], simple[6],
    ]
}

/// Compute the E7 Cartan matrix from simple roots.
/// For simply-laced (norm^2=2): C_ij = (s_i, s_j).
fn compute_cartan_matrix(roots: &[E7Root], si: &[usize; 7]) -> [[i32; 7]; 7] {
    let mut c = [[0i32; 7]; 7];
    for (c_row, &idx_i) in c.iter_mut().zip(si.iter()) {
        for (c_ij, &idx_j) in c_row.iter_mut().zip(si.iter()) {
            let ip = roots[idx_i].root.inner_product(&roots[idx_j].root);
            *c_ij = ip.round() as i32;
        }
    }
    c
}

/// Invert a 7x7 integer matrix via Gaussian elimination (returns f64).
#[allow(clippy::needless_range_loop)]
fn invert_7x7(m: [[i32; 7]; 7]) -> [[f64; 7]; 7] {
    let mut a = [[0.0f64; 7]; 7];
    let mut inv = [[0.0f64; 7]; 7];
    for i in 0..7 {
        for j in 0..7 {
            a[i][j] = f64::from(m[i][j]);
            inv[i][j] = if i == j { 1.0 } else { 0.0 };
        }
    }
    for col in 0..7 {
        let mut max_row = col;
        let mut max_val = a[col][col].abs();
        for row in (col + 1)..7 {
            if a[row][col].abs() > max_val {
                max_val = a[row][col].abs();
                max_row = row;
            }
        }
        assert!(max_val > 1e-10, "Cartan matrix is singular");
        if max_row != col {
            a.swap(col, max_row);
            inv.swap(col, max_row);
        }
        let pivot = a[col][col];
        for row in (col + 1)..7 {
            let factor = a[row][col] / pivot;
            for j in 0..7 {
                a[row][j] -= factor * a[col][j];
                inv[row][j] -= factor * inv[col][j];
            }
        }
    }
    for col in (0..7).rev() {
        let pivot = a[col][col];
        for j in 0..7 {
            inv[col][j] /= pivot;
        }
        for row in 0..col {
            let factor = a[row][col];
            for j in 0..7 {
                inv[row][j] -= factor * inv[col][j];
            }
        }
    }
    inv
}

/// Build the Tits bilinear form on simple roots.
///
/// f(s_i, s_j) encodes the extraspecial 2-cocycle:
/// - f(s_i, s_i) = 1
/// - f(s_i, s_j) = 1 for i < j if adjacent (Cartan entry != 0), else 0
/// - f(s_j, s_i) = (f(s_i, s_j) + C_ij) mod 2, ensuring
///   epsilon(a,b)*epsilon(b,a) = (-1)^{(a,b)}
fn build_tits_form(cartan: &[[i32; 7]; 7]) -> [[u8; 7]; 7] {
    let mut f = [[0u8; 7]; 7];
    for (i, f_row) in f.iter_mut().enumerate() {
        f_row[i] = 1;
    }
    // Cross-indexed: f[i][j] and f[j][i] require both indices
    #[allow(clippy::needless_range_loop)]
    for i in 0..7 {
        for j in (i + 1)..7 {
            let val = u8::from(cartan[i][j] != 0);
            f[i][j] = val;
            f[j][i] = ((i32::from(val) + cartan[i][j]).rem_euclid(2)) as u8;
        }
    }
    f
}

/// Decompose all roots into simple root basis coordinates.
/// Solves c = C^{-1} * b where b[j] = (root, s_j).
fn decompose_all_roots(
    roots: &[E7Root],
    simple_indices: &[usize; 7],
    cartan_inv: &[[f64; 7]; 7],
) -> HashMap<[i64; 8], [i32; 7]> {
    let mut decompositions = HashMap::new();
    for root in roots {
        let key = coord_key(&root.root.coords);
        let mut b = [0.0; 7];
        for (b_j, &si_j) in b.iter_mut().zip(simple_indices.iter()) {
            *b_j = root.root.inner_product(&roots[si_j].root);
        }
        let mut c = [0i32; 7];
        for (i, (c_i, inv_row)) in c.iter_mut().zip(cartan_inv.iter()).enumerate() {
            let val: f64 = inv_row
                .iter()
                .zip(b.iter())
                .map(|(&ci_j, &b_j)| ci_j * b_j)
                .sum();
            let rounded = val.round() as i32;
            assert!(
                (val - f64::from(rounded)).abs() < 0.01,
                "Non-integer root decomposition: coeff[{}] = {:.6} for root {:?}",
                i,
                val,
                root.root.coords
            );
            *c_i = rounded;
        }
        decompositions.insert(key, c);
    }
    decompositions
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extraspecial_cocycle_is_2_cocycle() {
        let basis = ChevalleyBasis::new();
        let f = &basis.tits_form;

        // Diagonal: f(s_i, s_i) = 1
        for (i, f_row) in f.iter().enumerate() {
            assert_eq!(f_row[i], 1, "f(s_{i}, s_{i}) should be 1");
        }

        // Cocycle condition: f(a,b) + f(b,a) = (a,b) mod 2
        // Cross-indexed: f[i][j] and f[j][i] require both indices
        let cartan = compute_cartan_matrix(&basis.roots, &basis.simple_root_indices);
        #[allow(clippy::needless_range_loop)]
        for i in 0..7 {
            for j in 0..7 {
                if i != j {
                    let sum = (i32::from(f[i][j]) + i32::from(f[j][i])).rem_euclid(2);
                    let expected = cartan[i][j].rem_euclid(2);
                    assert_eq!(
                        sum,
                        expected,
                        "Cocycle violated for ({i}, {j}): f+f'={sum}, C={c}",
                        c = cartan[i][j]
                    );
                }
            }
        }
    }

    #[test]
    fn test_chevalley_antisymmetry() {
        let basis = ChevalleyBasis::new();
        let roots = basis.roots();
        let mut nonzero_count = 0;

        for i in 0..roots.len() {
            for j in (i + 1)..roots.len() {
                let nab = basis.structure_constant(&roots[i], &roots[j]);
                let nba = basis.structure_constant(&roots[j], &roots[i]);
                assert!(
                    (nab + nba).abs() < 1e-14,
                    "Antisymmetry violated for roots {i} and {j}: N(a,b)={nab}, N(b,a)={nba}",
                );
                if nab.abs() > 0.5 {
                    nonzero_count += 1;
                }
            }
        }
        assert!(
            nonzero_count > 0,
            "Should find non-zero structure constants"
        );
    }

    #[test]
    fn test_jacobi_identity_exhaustive() {
        let basis = ChevalleyBasis::new();
        assert!(
            basis.verify_jacobi_identity(),
            "Jacobi identity must hold for all root triples"
        );
    }

    #[test]
    fn test_chevalley_values_are_unit() {
        let basis = ChevalleyBasis::new();
        let roots = basis.roots();
        for i in 0..roots.len() {
            for j in 0..roots.len() {
                let n = basis.structure_constant(&roots[i], &roots[j]);
                assert!(
                    n == 0.0 || n == 1.0 || n == -1.0,
                    "Structure constant must be 0, +1, or -1, got {n}",
                );
            }
        }
    }

    #[test]
    fn test_simple_root_structure_constants() {
        let basis = ChevalleyBasis::new();
        let cartan = compute_cartan_matrix(&basis.roots, &basis.simple_root_indices);

        for (i, (cartan_row, &si_idx)) in cartan
            .iter()
            .zip(basis.simple_root_indices.iter())
            .enumerate()
        {
            let si = &basis.roots[si_idx];
            for (j, (&c_ij, &sj_idx)) in cartan_row
                .iter()
                .zip(basis.simple_root_indices.iter())
                .enumerate()
            {
                if i == j {
                    continue;
                }
                let sj = &basis.roots[sj_idx];
                let n = basis.structure_constant(si, sj);
                if c_ij == -1 {
                    assert!(
                        n.abs() > 0.5,
                        "Adjacent simple roots {i},{j} should have |N|=1, got {n}",
                    );
                } else if c_ij == 0 {
                    assert_eq!(
                        n, 0.0,
                        "Non-adjacent simple roots {i},{j} should have N=0, got {n}",
                    );
                }
            }
        }
    }
}
