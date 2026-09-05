//! E8 lattice and root system.
//!
//! The E8 lattice is the unique even unimodular lattice in 8 dimensions.  It
//! contains 240 roots (vectors of squared length 2) which form the root system
//! of the exceptional Lie algebra `e_8`.
//!
//! # Construction
//!
//! Two equivalent presentations:
//! - `D_8` (integer points with even coordinate sum) plus half-integer translates.
//! - All `(x_1,...,x_8)` where either every `x_i` is integer with even sum,
//!   or every `x_i` is half-integer with even coordinate sum.
//!
//! # Literature
//! - Conway & Sloane, *Sphere Packings, Lattices and Groups* (3rd ed., 1998).
//! - Adams, *Lectures on Exceptional Lie Groups*.
//! - UOR Foundation (2024): *Atlas of Resonance Classes* (DOI: 10.5281/zenodo.17289540).
//!
//! # Numbering convention (resolved 2026-05-08)
//!
//! The Dynkin diagram is realised with the **branch at node 4**:
//!
//! ```text
//!   alpha_0 - alpha_1 - alpha_2 - alpha_3 - alpha_4 - alpha_5
//!                                          |
//!                                          alpha_6 - alpha_7
//! ```
//!
//! Three earlier conventions disagreed silently:
//! - The old `e8_simple_roots()` table (branch at node 4 -- correct).
//! - The old `e8_cartan_matrix()` (branch at node 2 -- a valid but *different*
//!   E8 numbering, internally consistent but **not** derivable from the simple
//!   roots).
//! - The Cartan in [`crate::lie::kac_moody`] (branch at node 4 -- correct,
//!   matching the simple roots).
//!
//! Both numberings are isomorphic E8s; the divergence was a labelling artefact.
//! The resolution adopts the branch-at-node-4 convention used by `kac_moody`
//! so the Cartan now strictly derives from the simple roots via
//! `A_ij = 2 (alpha_i, alpha_j) / (alpha_i, alpha_i)`. The cross-derivation
//! is asserted in `tests::cartan_matrix_derives_from_simple_roots`.

use std::collections::{HashMap, HashSet};

// ============================================================================
// Root vector and lattice type
// ============================================================================

/// An E8 root vector (8 components in standard Euclidean basis).
#[derive(Debug, Clone, PartialEq)]
pub struct E8Root {
    /// Components of the root vector.
    pub coords: [f64; 8],
    /// Squared length (should be 2 for valid roots).
    pub norm_sq: f64,
}

impl E8Root {
    /// Construct an E8 root from coordinates, computing `norm_sq` eagerly.
    pub fn new(coords: [f64; 8]) -> Self {
        let norm_sq: f64 = coords.iter().map(|x| x * x).sum();
        Self { coords, norm_sq }
    }

    /// True iff this vector has squared length `2.0` within `1e-10` tolerance.
    pub fn is_valid_root(&self) -> bool {
        (self.norm_sq - 2.0).abs() < 1e-10
    }

    /// Euclidean inner product with another root.
    pub fn inner_product(&self, other: &E8Root) -> f64 {
        self.coords
            .iter()
            .zip(other.coords.iter())
            .map(|(a, b)| a * b)
            .sum()
    }
}

/// The full E8 lattice with all 240 roots, simple basis, and Cartan matrix.
#[derive(Debug, Clone)]
pub struct E8Lattice {
    /// All 240 roots.
    pub roots: Vec<E8Root>,
    /// Simple roots (a basis of the root system).
    pub simple_roots: [E8Root; 8],
    /// Hardcoded Cartan matrix (8x8).
    pub cartan_matrix: [[i32; 8]; 8],
}

impl E8Lattice {
    /// Build a fresh `E8Lattice` with all standard data populated.
    pub fn new() -> Self {
        Self {
            roots: generate_e8_roots(),
            simple_roots: e8_simple_roots(),
            cartan_matrix: e8_cartan_matrix(),
        }
    }

    /// Number of lexicographically positive roots (first nonzero coord > 0).
    pub fn positive_root_count(&self) -> usize {
        self.positive_roots().len()
    }

    /// Roots whose first nonzero Euclidean coordinate is positive.
    pub fn positive_roots(&self) -> Vec<&E8Root> {
        self.roots
            .iter()
            .filter(|r| first_nonzero_positive(&r.coords))
            .collect()
    }

    /// Sanity-check the stored Cartan matrix: diagonal entries are 2 and
    /// the off-diagonal product `A_ij * A_ji` is in `{0, 1, 2, 3}` for each pair.
    pub fn verify_cartan_matrix(&self) -> bool {
        let c = &self.cartan_matrix;

        for (i, row) in c.iter().enumerate() {
            if row[i] != 2 {
                return false;
            }
        }

        for (i, row_i) in c.iter().enumerate() {
            for (j, &c_ij) in row_i.iter().enumerate() {
                if i != j && c_ij != 0 && c[j][i] != 0 {
                    let prod = c_ij * c[j][i];
                    if !(0..=3).contains(&prod) {
                        return false;
                    }
                }
            }
        }

        true
    }

    /// All roots whose inner product with `ref_root` rounds to `target_ip`.
    pub fn roots_at_distance(&self, ref_root: &E8Root, target_ip: i32) -> Vec<&E8Root> {
        self.roots
            .iter()
            .filter(|r| (r.inner_product(ref_root).round() as i32) == target_ip)
            .collect()
    }

    /// Truncated theta function `Theta_E8(q) = 1 + sum count(n_sq) * q^{n_sq}`.
    ///
    /// E8's theta function is a modular form of weight 4 for `SL(2, Z)`.
    pub fn compute_theta_function(&self, q: f64, max_norm_sq: usize) -> f64 {
        if q.abs() >= 1.0 {
            return f64::INFINITY;
        }

        let mut theta = 1.0;
        for n_sq in (2..=max_norm_sq).step_by(2) {
            let count = self.count_vectors_with_norm_sq(n_sq);
            theta += count as f64 * q.powi(n_sq as i32);
        }
        theta
    }

    /// Count lattice vectors with the given squared norm.
    ///
    /// For E8: `240 * sigma_3(n)` where `n = norm_sq / 2` and `sigma_3(n)` is
    /// the sum of cubes of divisors of `n`.
    pub fn count_vectors_with_norm_sq(&self, norm_sq: usize) -> usize {
        if !norm_sq.is_multiple_of(2) {
            return 0;
        }
        let n = norm_sq / 2;
        if n == 0 {
            return 1;
        }
        let mut sigma3 = 0;
        for d in 1..=n {
            if n.is_multiple_of(d) {
                sigma3 += d * d * d;
            }
        }
        240 * sigma3
    }

    /// Sphere-packing density of E8 (Viazovska 2016 optimum in `R^8`).
    ///
    /// `Delta = V_8(r) / det(L)` with `r = sqrt(2)/2`, `V_8(r) = (pi^4/24) r^8`,
    /// and `det(E8) = 1` (unimodular). Result: `pi^4 / 384 ~= 0.25367`.
    pub fn sphere_packing_density(&self) -> f64 {
        let r = 1.0 / 2.0_f64.sqrt();
        let vol_8d_sphere = (std::f64::consts::PI.powi(4) / 24.0) * r.powi(8);
        let lattice_vol = 1.0;
        vol_8d_sphere / lattice_vol
    }
}

impl Default for E8Lattice {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Free functions: roots, simple basis, Cartan, Weyl, fundamental weights
// ============================================================================

/// Generate all 240 E8 roots.
///
/// - 112 roots of type `(+/-1, +/-1, 0, 0, 0, 0, 0, 0)` (and permutations).
/// - 128 roots of type `(+/-1/2)^8` with an even number of minus signs.
pub fn generate_e8_roots() -> Vec<E8Root> {
    let mut roots = Vec::with_capacity(240);

    for i in 0..8 {
        for j in (i + 1)..8 {
            for sign_i in [-1.0, 1.0] {
                for sign_j in [-1.0, 1.0] {
                    let mut coords = [0.0; 8];
                    coords[i] = sign_i;
                    coords[j] = sign_j;
                    roots.push(E8Root::new(coords));
                }
            }
        }
    }

    for sign_pattern in 0u8..=255 {
        let minus_count = sign_pattern.count_ones();
        if minus_count.is_multiple_of(2) {
            let mut coords = [0.5; 8];
            for (bit, coord) in coords.iter_mut().enumerate() {
                if (sign_pattern >> bit) & 1 == 1 {
                    *coord = -0.5;
                }
            }
            roots.push(E8Root::new(coords));
        }
    }

    assert_eq!(roots.len(), 240, "E8 should have exactly 240 roots");
    roots
}

/// E8 Cartan matrix with branch at node 4.
///
/// The Dynkin diagram is `0 - 1 - 2 - 3 - 4 - 5` with `4 - 6 - 7` branching
/// off node 4. This matches both [`e8_simple_roots`] (the matrix derives from
/// the inner products of those vectors) and [`crate::lie::kac_moody`]'s E8
/// Cartan, used to seed the affine and hyperbolic extensions `E_9, E_10, E_11`.
pub fn e8_cartan_matrix() -> [[i32; 8]; 8] {
    [
        [2, -1, 0, 0, 0, 0, 0, 0],
        [-1, 2, -1, 0, 0, 0, 0, 0],
        [0, -1, 2, -1, 0, 0, 0, 0],
        [0, 0, -1, 2, -1, 0, 0, 0],
        [0, 0, 0, -1, 2, -1, -1, 0], // branch at node 4 (degree 3)
        [0, 0, 0, 0, -1, 2, 0, 0],
        [0, 0, 0, 0, -1, 0, 2, -1],
        [0, 0, 0, 0, 0, 0, -1, 2],
    ]
}

/// Eight simple roots with branch at node 4.
///
/// `alpha_0..alpha_5` form a length-6 chain in the Euclidean basis; `alpha_6`
/// branches off `alpha_4` into the half-integer leaf `alpha_7`. The Cartan
/// matrix [`e8_cartan_matrix`] is derived from these via
/// `A_ij = 2 (alpha_i, alpha_j) / (alpha_i, alpha_i)` -- see
/// `tests::cartan_matrix_derives_from_simple_roots`.
pub fn e8_simple_roots() -> [E8Root; 8] {
    [
        E8Root::new([1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        E8Root::new([0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        E8Root::new([0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0]),
        E8Root::new([0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0]),
        E8Root::new([0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0]),
        E8Root::new([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0]),
        E8Root::new([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0]),
        E8Root::new([-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5]),
    ]
}

/// Order of the Weyl group `W(E8) = 696,729,600 = 2^14 * 3^5 * 5^2 * 7`.
pub fn e8_weyl_group_order() -> u64 {
    696_729_600
}

/// Histogram of pairwise inner products across a root list.
///
/// Returns `(rounded_inner_product, count)` pairs sorted ascending. For E8 the
/// support is `{-2, -1, 0, 1}` (the value `+2` arises only at `i == j` and is
/// excluded by the `i < j` iteration).
pub fn compute_e8_inner_products(roots: &[E8Root]) -> Vec<(i32, usize)> {
    let mut counts: HashMap<i32, usize> = HashMap::new();

    for i in 0..roots.len() {
        for j in (i + 1)..roots.len() {
            let ip = roots[i].inner_product(&roots[j]).round() as i32;
            *counts.entry(ip).or_insert(0) += 1;
        }
    }

    let mut result: Vec<_> = counts.into_iter().collect();
    result.sort_by_key(|(ip, _)| *ip);
    result
}

/// Fundamental weights (dual basis to the simple roots).
pub fn e8_fundamental_weights() -> [[f64; 8]; 8] {
    [
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 0.0],
        [3.0, 3.0, 3.0, 3.0, 2.0, 1.0, 1.0, 0.0],
        [4.0, 4.0, 4.0, 3.0, 2.0, 1.0, 1.0, 0.0],
        [5.0, 4.0, 3.0, 2.0, 2.0, 1.0, 1.0, 0.0],
        [4.0, 3.0, 2.0, 2.0, 2.0, 1.0, 1.0, 0.0],
        [2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
    .map(|mut w| {
        for val in &mut w {
            *val *= 0.5;
        }
        w
    })
}

/// Simple reflection `s_alpha(v) = v - 2 (v,alpha)/(alpha,alpha) alpha`.
/// For E8 roots, `(alpha,alpha) = 2`, so this is `v - (v,alpha) alpha`.
/// Returns `None` unless both inputs are exact roots in the stored E8 convention.
pub fn weyl_reflect(v: &E8Root, alpha: &E8Root) -> Option<E8Root> {
    if !is_standard_e8_root(v) || !is_standard_e8_root(alpha) {
        return None;
    }
    let ip = v.inner_product(alpha);
    let coords = std::array::from_fn(|index| v.coords[index] - ip * alpha.coords[index]);
    Some(E8Root::new(coords))
}

fn is_standard_e8_root(root: &E8Root) -> bool {
    if root.norm_sq != 2.0 {
        return false;
    }
    let integer = root
        .coords
        .iter()
        .all(|&coordinate| matches!(coordinate, -1.0 | 0.0 | 1.0))
        && root
            .coords
            .iter()
            .filter(|&&coordinate| coordinate != 0.0)
            .count()
            == 2;
    let spinor = root.coords.iter().all(|coordinate| coordinate.abs() == 0.5)
        && root
            .coords
            .iter()
            .filter(|&&coordinate| coordinate < 0.0)
            .count()
            .is_multiple_of(2);
    integer || spinor
}

fn quantize_root(coords: &[f64; 8]) -> [i8; 8] {
    coords.map(|x| (2.0 * x).round() as i8)
}

/// Generate all 240 E8 roots as the Weyl orbit of the eight simple roots.
pub fn generate_e8_roots_by_weyl() -> Vec<E8Root> {
    let simple = e8_simple_roots();
    let mut seen: HashSet<[i8; 8]> = HashSet::new();
    let mut roots: Vec<E8Root> = Vec::with_capacity(240);
    let mut stack: Vec<E8Root> = Vec::new();
    for s in &simple {
        let q = quantize_root(&s.coords);
        if seen.insert(q) {
            roots.push(s.clone());
            stack.push(s.clone());
        }
    }
    while let Some(v) = stack.pop() {
        for alpha in &simple {
            let w = weyl_reflect(&v, alpha).expect("Weyl closure contains exact E8 roots");
            let q = quantize_root(&w.coords);
            if seen.insert(q) {
                stack.push(w.clone());
                roots.push(w);
            }
        }
    }
    roots
}

/// Expand `root` in the simple-root basis: `root = sum c_i alpha_i`.
/// Integer for every root in the Weyl orbit of [`e8_simple_roots`].
/// Returns `None` for a vector outside the exact stored E8 root system.
pub fn simple_coordinates(root: &E8Root) -> Option<[i32; 8]> {
    if !is_standard_e8_root(root) {
        return None;
    }
    let simple = e8_simple_roots();
    let mut m = nalgebra::SMatrix::<f64, 8, 8>::zeros();
    for (j, alpha) in simple.iter().enumerate() {
        for (i, &x) in alpha.coords.iter().enumerate() {
            m[(i, j)] = x;
        }
    }
    let rhs = nalgebra::SVector::<f64, 8>::from(root.coords);
    let c = m
        .lu()
        .solve(&rhs)
        .expect("simple roots of E8 are a basis of R^8");
    Some([0, 1, 2, 3, 4, 5, 6, 7].map(|i| c[i].round() as i32))
}

fn first_nonzero_positive(coords: &[f64; 8]) -> bool {
    for &x in coords {
        if x > 1e-10 {
            return true;
        }
        if x < -1e-10 {
            return false;
        }
    }
    false
}

/// Highest root of the Weyl orbit of the stored simple system (height 29).
pub fn weyl_highest_root() -> E8Root {
    generate_e8_roots_by_weyl()
        .into_iter()
        .max_by_key(|root| height(root).expect("Weyl closure contains exact E8 roots"))
        .expect("Weyl orbit of E8 simple roots is nonempty")
}

/// Height: sum of simple-root coefficients.
/// Returns `None` for a vector outside the exact stored E8 root system.
pub fn height(root: &E8Root) -> Option<i32> {
    simple_coordinates(root).map(|coordinates| coordinates.iter().sum())
}

/// 112 D8-type roots (two ±1, six 0) vs 128 half-integer spinor roots.
/// Returns `None` if any supplied vector is outside the exact E8 root system.
pub fn e8_root_type_counts(roots: &[E8Root]) -> Option<(usize, usize)> {
    let mut d8 = 0usize;
    let mut spinor = 0usize;
    for r in roots {
        if !is_standard_e8_root(r) {
            return None;
        }
        let half = r.coords.iter().any(|x| (x.abs() - 0.5).abs() < 1e-9);
        if half {
            spinor += 1;
        } else {
            d8 += 1;
        }
    }
    Some((d8, spinor))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e8_has_exactly_240_roots() {
        let roots = generate_e8_roots();
        assert_eq!(roots.len(), 240);
    }

    #[test]
    fn every_root_has_norm_squared_two() {
        for root in &generate_e8_roots() {
            assert!(
                root.is_valid_root(),
                "root {:?} has norm^2 = {}",
                root.coords,
                root.norm_sq
            );
        }
    }

    #[test]
    fn each_simple_root_is_a_valid_root() {
        for (i, root) in e8_simple_roots().iter().enumerate() {
            assert!(root.is_valid_root(), "simple root {} has invalid norm", i);
        }
    }

    #[test]
    fn cartan_matrix_passes_basic_structural_checks() {
        let lattice = E8Lattice::new();
        assert!(lattice.verify_cartan_matrix());
    }

    #[test]
    fn pairwise_inner_products_lie_in_minus_two_to_two() {
        let roots = generate_e8_roots();
        for (ip, _) in compute_e8_inner_products(&roots) {
            assert!((-2..=2).contains(&ip), "unexpected inner product {ip}");
        }
    }

    #[test]
    fn weyl_group_order_factorizes_correctly() {
        let order = e8_weyl_group_order();
        assert_eq!(order, 696_729_600);
        let expected = 2u64.pow(14) * 3u64.pow(5) * 5u64.pow(2) * 7;
        assert_eq!(order, expected);
    }

    #[test]
    fn positive_root_count_is_half_of_total() {
        assert_eq!(E8Lattice::new().positive_root_count(), 120);
    }

    #[test]
    fn sphere_packing_density_matches_pi_fourth_over_384() {
        let lattice = E8Lattice::new();
        let expected = std::f64::consts::PI.powi(4) / 384.0;
        let actual = lattice.sphere_packing_density();
        assert!(
            (actual - expected).abs() < 1e-12,
            "density {actual} != pi^4/384 = {expected}"
        );
    }

    /// The Cartan matrix derives from the simple roots via
    /// `A_ij = 2 (alpha_i, alpha_j) / (alpha_i, alpha_i)`. This was previously
    /// `#[ignore]`d due to a node-numbering divergence (branch at 2 vs branch
    /// at 4); resolved 2026-05-08 by aligning to branch-at-4.
    #[test]
    fn cartan_matrix_derives_from_simple_roots() {
        let simple = e8_simple_roots();
        let cartan = e8_cartan_matrix();

        for (i, alpha_i) in simple.iter().enumerate() {
            let dot_ii = alpha_i.inner_product(alpha_i);
            for (j, alpha_j) in simple.iter().enumerate() {
                let dot_ij = alpha_i.inner_product(alpha_j);
                let derived = (2.0 * dot_ij / dot_ii).round() as i32;
                assert_eq!(
                    derived, cartan[i][j],
                    "mismatch at ({i},{j}): derived={derived}, hardcoded={}",
                    cartan[i][j]
                );
            }
        }
    }

    #[test]
    fn weyl_orbit_of_simple_roots_is_240_distinct_norm_two() {
        let weyl = generate_e8_roots_by_weyl();
        assert_eq!(weyl.len(), 240);
        let set: HashSet<[i8; 8]> = weyl.iter().map(|r| quantize_root(&r.coords)).collect();
        assert_eq!(set.len(), 240);
        let combinatorial: HashSet<_> = generate_e8_roots()
            .iter()
            .map(|root| quantize_root(&root.coords))
            .collect();
        assert_eq!(set, combinatorial);
        for simple in e8_simple_roots() {
            assert!(combinatorial.contains(&quantize_root(&simple.coords)));
        }
        for r in &weyl {
            assert!(r.is_valid_root());
        }
        let (d8, spinor) = e8_root_type_counts(&weyl).unwrap();
        assert_eq!((d8, spinor), (112, 128));
    }

    #[test]
    fn simple_roots_have_height_one_and_reconstruct() {
        let simple = e8_simple_roots();
        for (i, alpha) in simple.iter().enumerate() {
            let c = simple_coordinates(alpha).unwrap();
            assert_eq!(height(alpha), Some(1));
            for (j, &cj) in c.iter().enumerate() {
                let expect = if i == j { 1 } else { 0 };
                assert_eq!(cj, expect, "simple root {i} coord {j} = {cj}");
            }
        }
    }

    #[test]
    fn every_root_reconstructs_from_simple_coordinates() {
        let simple = e8_simple_roots();
        for root in &generate_e8_roots_by_weyl() {
            let c = simple_coordinates(root).unwrap();
            let mut rec = [0.0_f64; 8];
            for (j, alpha) in simple.iter().enumerate() {
                for i in 0..8 {
                    rec[i] += f64::from(c[j]) * alpha.coords[i];
                }
            }
            for i in 0..8 {
                assert!(
                    (rec[i] - root.coords[i]).abs() < 1e-8,
                    "recon mismatch at {i}: {} vs {}",
                    rec[i],
                    root.coords[i]
                );
            }
        }
    }

    #[test]
    fn highest_root_has_height_29_and_120_positive_roots() {
        let lattice = E8Lattice::new();
        assert_eq!(height(&weyl_highest_root()), Some(29));
        assert_eq!(lattice.positive_roots().len(), 120);
        let weyl = generate_e8_roots_by_weyl();
        assert_eq!(
            weyl.iter()
                .filter(|r| height(r).is_some_and(|value| value > 0))
                .count(),
            120
        );
        let (d8, spinor) = e8_root_type_counts(&lattice.roots).unwrap();
        assert_eq!((d8, spinor), (112, 128));
    }

    #[test]
    fn reflecting_a_simple_root_through_itself_negates_it() {
        let alpha = &e8_simple_roots()[3];
        let neg = weyl_reflect(alpha, alpha).unwrap();
        for i in 0..8 {
            assert!((neg.coords[i] + alpha.coords[i]).abs() < 1e-10);
        }
        assert_eq!(height(&neg), Some(-1));
    }

    #[test]
    fn checked_root_helpers_reject_nonroots_and_stale_norms() {
        let mut stale_norm = e8_simple_roots()[0].clone();
        stale_norm.norm_sq = 0.0;
        let invalid = [
            E8Root::new([0.0; 8]),
            E8Root::new([f64::NAN; 8]),
            E8Root::new([f64::INFINITY; 8]),
            E8Root::new([2.0_f64.sqrt(), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            E8Root::new([-0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
            stale_norm,
        ];
        let valid = &e8_simple_roots()[0];
        for root in invalid {
            assert!(weyl_reflect(&root, valid).is_none());
            assert!(weyl_reflect(valid, &root).is_none());
            assert!(simple_coordinates(&root).is_none());
            assert!(height(&root).is_none());
            assert!(e8_root_type_counts(&[root]).is_none());
        }
    }

    /// The E8 Cartan in this module must match the E8 Cartan in
    /// [`crate::lie::kac_moody`] entry-by-entry, so the affine/hyperbolic
    /// extensions `E_9, E_10, E_11` see a consistent base.
    #[test]
    fn cartan_matrix_matches_kac_moody_e8() {
        let ours = e8_cartan_matrix();
        let kac_moody_e8 = crate::lie::kac_moody::e8_cartan();
        assert_eq!(kac_moody_e8.rank(), 8);
        for i in 0..8 {
            for j in 0..8 {
                assert_eq!(
                    ours[i][j],
                    kac_moody_e8.get(i, j),
                    "Cartan mismatch at ({i},{j}) between e8/root_system and kac_moody",
                );
            }
        }
    }
}
