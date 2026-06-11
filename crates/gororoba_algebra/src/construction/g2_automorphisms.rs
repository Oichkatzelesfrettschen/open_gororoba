//! G2 Automorphism Group of the Octonions.
//!
//! G2 is the smallest exceptional simple Lie group (14-dimensional). It arises as
//! Aut(O), the automorphism group of the octonion algebra: the set of all linear
//! maps f: O -> O satisfying f(xy) = f(x)f(y) for all x, y in O.
//!
//! Key facts:
//! - dim(G2) = 14 = dim(SO(7)) - 7 (codimension in SO(7))
//! - G2 acts on Im(O) = R^7 as a subgroup of SO(7)
//! - G2 preserves the 3-form phi(x,y,z) = <x, yz> on Im(O)
//! - G2 preserves the cross product on Im(O): x * y = `[x,y]`/2
//! - Lie algebra: g2 = Der(O), the derivation algebra of the octonions
//! - A derivation D satisfies D(xy) = D(x)y + xD(y) (Leibniz rule)
//!
//! Implementation approach: Der(O) is computed as the null space of the Leibniz
//! constraint system. A derivation is a 7x7 skew-symmetric matrix on Im(O)
//! (21 parameters in so(7)) satisfying D(e_i*e_j) = D(e_i)*e_j + e_i*D(e_j).
//! These constraints reduce the 21 parameters to exactly 14 = dim(g2).
//!
//! Reference: Baez "The Octonions" (2002), Schafer "Non-Associative Algebras" (1966),
//! Harvey "Spinors and Calibrations" (1990)

use super::octonion::Octonion;

/// Build the 21 upper-triangle index pairs (i, j) with i < j for a 7x7
/// skew-symmetric matrix parameterizing so(7).
///
/// Each pair (i, j) corresponds to one free parameter in Im(O) coordinates
/// (0-indexed, i.e., 0..6). The ordering is row-by-row.
fn build_skew_param_pairs() -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    for i in 0..7 {
        for j in (i + 1)..7 {
            pairs.push((i, j));
        }
    }
    pairs
}

/// Construct the 8x8 matrix basis element for the k-th so(7) parameter.
///
/// The matrix has +1 at (a+1, b+1) and -1 at (b+1, a+1) where (a, b) is the
/// k-th pair from `param_pairs`. The +1 offset accounts for the scalar slot
/// in the full octonion 8-vector.
fn build_param_matrix(a: usize, b: usize) -> [[f64; 8]; 8] {
    let mut m = [[0.0f64; 8]; 8];
    m[a + 1][b + 1] = 1.0;
    m[b + 1][a + 1] = -1.0;
    m
}

/// Compute the contribution of one parameter matrix `pm` to the Leibniz
/// constraint row for output component `c` on basis pair (ei, ej).
///
/// Returns `lhs_c - rhs_c` where:
///   lhs_c = (pm * (ei*ej))`[c]`
///   rhs_c = ((pm*ei)*ej)`[c]` + (ei*(pm*ej))`[c]`
fn leibniz_row_entry(
    pm: &[[f64; 8]; 8],
    ei: &Octonion,
    ej: &Octonion,
    ei_ej: &Octonion,
    octonion_i_idx: usize,
    octonion_j_idx: usize,
    c: usize,
) -> f64 {
    // LHS: D(ei * ej) component c, where D is the matrix pm.
    let lhs_c: f64 = pm[c]
        .iter()
        .zip(ei_ej.components.iter())
        .map(|(&pm_cs, &comp_s)| pm_cs * comp_s)
        .sum();

    // D(ei) = column octonion_i_idx of pm (pm acts on the left).
    let mut d_ei_arr = [0.0f64; 8];
    for (s, slot) in d_ei_arr.iter_mut().enumerate() {
        *slot = pm[s][octonion_i_idx];
    }
    let d_ei_oct = Octonion::new(d_ei_arr);
    let d_ei_ej = d_ei_oct.multiply(ej);

    // D(ej) = column octonion_j_idx of pm.
    let mut d_ej_arr = [0.0f64; 8];
    for (s, slot) in d_ej_arr.iter_mut().enumerate() {
        *slot = pm[s][octonion_j_idx];
    }
    let d_ej_oct = Octonion::new(d_ej_arr);
    let ei_d_ej = ei.multiply(&d_ej_oct);

    let rhs_c = d_ei_ej.components[c] + ei_d_ej.components[c];
    lhs_c - rhs_c
}

/// Build all non-trivial Leibniz constraint rows for the 21-parameter so(7) basis.
///
/// For each basis pair (ei, ej) with i <= j in {1..7} and each output component
/// c in {0..7}, the Leibniz condition D(ei*ej) = D(ei)*ej + ei*D(ej) yields one
/// linear equation in the 21 parameters. Rows whose Euclidean norm is below 1e-14
/// are discarded as numerically trivial.
fn build_leibniz_constraint_rows(param_matrices: &[[[f64; 8]; 8]]) -> Vec<[f64; 21]> {
    let mut constraint_rows: Vec<[f64; 21]> = Vec::new();

    for i in 1..8usize {
        for j in i..8usize {
            let ei = Octonion::basis(i);
            let ej = Octonion::basis(j);
            let ei_ej = ei.multiply(&ej);

            for c in 0..8usize {
                let mut row = [0.0f64; 21];
                for (k, pm) in param_matrices.iter().enumerate() {
                    row[k] = leibniz_row_entry(pm, &ei, &ej, &ei_ej, i, j, c);
                }
                let norm: f64 = row.iter().map(|x| x * x).sum::<f64>().sqrt();
                if norm > 1e-14 {
                    constraint_rows.push(row);
                }
            }
        }
    }

    constraint_rows
}

/// Perform Gaussian elimination with partial pivoting on a constraint matrix
/// (n_constraints rows, n_params=21 columns).
///
/// Returns `(pivot_cols, reduced_matrix)` where `pivot_cols` is the ordered
/// list of pivot column indices and `reduced_matrix` is the fully reduced
/// (reduced row echelon) form.
fn gaussian_eliminate_partial_pivot(
    constraint_rows: &[[f64; 21]],
    n_params: usize,
) -> (Vec<usize>, Vec<Vec<f64>>) {
    let n_constraints = constraint_rows.len();
    let mut matrix: Vec<Vec<f64>> = constraint_rows.iter().map(|row| row.to_vec()).collect();

    let mut pivot_cols: Vec<usize> = Vec::new();
    let mut current_row = 0usize;

    for col in 0..n_params {
        // Find the row with maximum absolute value in this column.
        let mut max_val = 0.0f64;
        let mut max_row = current_row;
        for (row, mat_row) in matrix.iter().enumerate().skip(current_row) {
            if mat_row[col].abs() > max_val {
                max_val = mat_row[col].abs();
                max_row = row;
            }
        }
        if max_val < 1e-12 {
            continue; // Column is a free variable; skip.
        }

        matrix.swap(current_row, max_row);
        pivot_cols.push(col);

        let pivot = matrix[current_row][col];
        for row in 0..n_constraints {
            if row == current_row {
                continue;
            }
            let factor = matrix[row][col] / pivot;
            // Gauss-Jordan row reduction: matrix[row][c] depends on
            // matrix[current_row][c]; needs `c` to index both rows.
            #[allow(clippy::needless_range_loop)]
            for c in 0..n_params {
                matrix[row][c] -= factor * matrix[current_row][c];
            }
        }
        current_row += 1;
    }

    (pivot_cols, matrix)
}

/// Extract a basis for the null space of the reduced constraint matrix.
///
/// For each free column (not in `pivot_cols`), set that parameter to 1.0,
/// back-substitute to resolve the pivot variables, and collect the resulting
/// 21-element vector.
fn extract_null_basis(
    reduced_matrix: &[Vec<f64>],
    pivot_cols: &[usize],
    n_params: usize,
) -> Vec<[f64; 21]> {
    let free_cols: Vec<usize> = (0..n_params).filter(|c| !pivot_cols.contains(c)).collect();

    let mut null_basis: Vec<[f64; 21]> = Vec::new();
    for &free_col in &free_cols {
        let mut vec = [0.0f64; 21];
        vec[free_col] = 1.0;

        for (pivot_idx, &pivot_col) in pivot_cols.iter().enumerate().rev() {
            let pivot_val = reduced_matrix[pivot_idx][pivot_col];
            if pivot_val.abs() < 1e-14 {
                continue;
            }
            let mut sum = 0.0f64;
            for c in 0..n_params {
                if c != pivot_col {
                    sum += reduced_matrix[pivot_idx][c] * vec[c];
                }
            }
            vec[pivot_col] = -sum / pivot_val;
        }
        null_basis.push(vec);
    }

    null_basis
}

/// A derivation of the octonion algebra: a linear map D: O -> O
/// satisfying the Leibniz rule D(xy) = D(x)y + xD(y).
///
/// Represented as an 8x8 matrix acting on octonion components.
/// First row and column are zero (D maps scalars to zero).
/// The 7x7 imaginary block is skew-symmetric (D lies in so(7)).
/// The derivation algebra Der(O) is isomorphic to g2.
#[derive(Clone, Debug)]
pub struct OctonionDerivation {
    /// 8x8 matrix representation of the derivation
    pub matrix: [[f64; 8]; 8],
}

impl OctonionDerivation {
    /// Apply the derivation to an octonion.
    pub fn apply(&self, x: &Octonion) -> Octonion {
        let mut result = [0.0; 8];
        for (i, res) in result.iter_mut().enumerate() {
            for j in 0..8 {
                *res += self.matrix[i][j] * x.components[j];
            }
        }
        Octonion::new(result)
    }

    /// Verify the Leibniz rule: D(xy) = D(x)y + xD(y) for given x, y.
    pub fn verify_leibniz(&self, x: &Octonion, y: &Octonion, tol: f64) -> bool {
        let xy = x.multiply(y);
        let d_xy = self.apply(&xy);

        let dx = self.apply(x);
        let dy = self.apply(y);
        let dx_y = dx.multiply(y);
        let x_dy = x.multiply(&dy);
        let rhs = dx_y.add(&x_dy);

        let diff: f64 = d_xy
            .components
            .iter()
            .zip(rhs.components.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        diff < tol
    }

    /// Matrix norm (Frobenius).
    pub fn frobenius_norm(&self) -> f64 {
        let mut sum = 0.0;
        for row in &self.matrix {
            for &val in row {
                sum += val * val;
            }
        }
        sum.sqrt()
    }

    /// Check if this derivation maps scalars to zero: D(1) = 0.
    pub fn maps_scalars_to_zero(&self, tol: f64) -> bool {
        let one = Octonion::unit();
        let d_one = self.apply(&one);
        d_one.norm() < tol
    }

    /// Check if this derivation is skew-symmetric: <D(x), y> + <x, D(y)> = 0.
    pub fn verify_antisymmetry(&self, x: &Octonion, y: &Octonion, tol: f64) -> bool {
        let dx = self.apply(x);
        let dy = self.apply(y);
        let lhs = dx.inner_product(y) + x.inner_product(&dy);
        lhs.abs() < tol
    }
}

/// Compute a basis for Der(O) = g2 by solving the Leibniz constraint system.
///
/// A derivation D is a skew-symmetric 7x7 matrix on Im(O), embedded as 8x8
/// with zero first row/column. The skew-symmetric condition gives 21 free
/// parameters (upper triangle of 7x7). The Leibniz rule D(e_i*e_j) = D(e_i)*e_j + e_i*D(e_j)
/// provides additional linear constraints, reducing the space to 14 = dim(g2).
///
/// Returns a basis of OctonionDerivation objects spanning g2.
pub fn compute_g2_basis() -> Vec<OctonionDerivation> {
    const N_PARAMS: usize = 21;

    // Step 1: Enumerate the 21 free parameters of so(7) as (i,j) pairs.
    let param_pairs = build_skew_param_pairs();
    assert_eq!(param_pairs.len(), N_PARAMS);

    // Step 2: One 8x8 basis matrix per free parameter.
    let param_matrices: Vec<[[f64; 8]; 8]> = param_pairs
        .iter()
        .map(|&(a, b)| build_param_matrix(a, b))
        .collect();

    // Step 3: Collect linear Leibniz constraints A*p = 0 (one row per
    // non-trivial component equation over all basis pairs (ei, ej)).
    let constraint_rows = build_leibniz_constraint_rows(&param_matrices);

    // Step 4: Reduce to row echelon form; pivot columns identify the rank.
    let (pivot_cols, reduced_matrix) = gaussian_eliminate_partial_pivot(&constraint_rows, N_PARAMS);
    let rank = pivot_cols.len();
    let null_dim = N_PARAMS - rank;

    // Step 5: Extract one null-space vector per free variable.
    let null_basis = extract_null_basis(&reduced_matrix, &pivot_cols, N_PARAMS);

    // Verify we got the expected dimension before converting.
    assert_eq!(
        null_dim,
        14,
        "Der(O) should have dimension 14, got {} (rank={}, constraints={})",
        null_dim,
        rank,
        constraint_rows.len()
    );

    // Step 6: Convert null-space vectors to OctonionDerivation objects.
    null_basis
        .iter()
        .map(|null_vec| {
            let mut m = [[0.0f64; 8]; 8];
            for (k, &coeff) in null_vec.iter().enumerate() {
                let (a, b) = param_pairs[k];
                m[a + 1][b + 1] += coeff;
                m[b + 1][a + 1] -= coeff;
            }
            OctonionDerivation { matrix: m }
        })
        .collect()
}

/// G2 structure computations on the imaginary octonions Im(O) = R^7.
///
/// G2 is characterized as the subgroup of GL(7,R) preserving:
/// 1. The inner product on Im(O)
/// 2. The cross product: x * y = `[x,y]`/2
/// 3. The associative 3-form: phi(x,y,z) = <x, yz>
pub struct G2Structure;

impl G2Structure {
    /// Dimension of the G2 Lie group.
    pub fn dimension() -> usize {
        14
    }

    /// Rank of G2 (dimension of maximal torus).
    pub fn rank() -> usize {
        2
    }

    /// Dimension of the fundamental representation (Im(O) = R^7).
    pub fn fundamental_rep_dim() -> usize {
        7
    }

    /// Dimension of the adjoint representation (g2 itself).
    pub fn adjoint_rep_dim() -> usize {
        14
    }

    /// Compute the associative 3-form: phi(x, y, z) = <x, yz>.
    /// This is the calibration form preserved by G2.
    pub fn three_form(x: &Octonion, y: &Octonion, z: &Octonion) -> f64 {
        let yz = y.multiply(z);
        x.inner_product(&yz)
    }

    /// Verify the 3-form is alternating: phi(x,y,z) = -phi(y,x,z).
    pub fn verify_three_form_alternating(
        x: &Octonion,
        y: &Octonion,
        z: &Octonion,
        tol: f64,
    ) -> bool {
        let phi_xyz = Self::three_form(x, y, z);
        let phi_yxz = Self::three_form(y, x, z);
        (phi_xyz + phi_yxz).abs() < tol
    }

    /// Verify cross product preservation by a linear map.
    pub fn verify_cross_product_preserved(
        f: &dyn Fn(&Octonion) -> Octonion,
        x: &Octonion,
        y: &Octonion,
        tol: f64,
    ) -> bool {
        let cp_xy = x.cross_product(y);
        let f_cp = f(&cp_xy);

        let fx = f(x);
        let fy = f(y);
        let cp_fxfy = fx.cross_product(&fy);

        let diff: f64 = f_cp
            .components
            .iter()
            .zip(cp_fxfy.components.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        diff < tol
    }

    /// Verify inner product reconstruction from cross product on Im(O):
    /// <a, b> = (1/6) sum_{k=1}^{7} <a x e_k, b x e_k>
    /// Valid for purely imaginary octonions a, b.
    pub fn verify_inner_product_from_cross(a: &Octonion, b: &Octonion, tol: f64) -> bool {
        let ip_direct = a.inner_product(b);

        let mut trace_sum = 0.0;
        for k in 1..8 {
            let ek = Octonion::basis(k);
            let a_cross_ek = a.cross_product(&ek);
            let b_cross_ek = b.cross_product(&ek);
            trace_sum += a_cross_ek.inner_product(&b_cross_ek);
        }
        let ip_from_cross = trace_sum / 6.0;

        (ip_direct - ip_from_cross).abs() < tol
    }

    /// Verify dimension via two independent routes:
    /// Route 1: G2 acts transitively on S^6, stabilizer = SU(3).
    ///          dim(G2) = dim(S^6) + dim(SU(3)) = 6 + 8 = 14.
    /// Route 2: G2 embeds in SO(7), codimension = dim(SO(7)/G2) = 7.
    ///          dim(G2) = dim(SO(7)) - 7 = 21 - 7 = 14.
    pub fn verify_dimension_formula() -> bool {
        let dim_so7 = 7 * 6 / 2; // 21
        let codim_in_so7 = 7;
        let route2 = dim_so7 - codim_in_so7; // 14

        let dim_s6 = 6;
        let dim_su3 = 8;
        let route1 = dim_s6 + dim_su3; // 14

        route1 == 14 && route2 == 14
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_g2_dimension() {
        assert_eq!(G2Structure::dimension(), 14);
        assert_eq!(G2Structure::rank(), 2);
        assert_eq!(G2Structure::fundamental_rep_dim(), 7);
        assert_eq!(G2Structure::adjoint_rep_dim(), 14);
    }

    #[test]
    fn test_g2_dimension_formula() {
        assert!(G2Structure::verify_dimension_formula());
    }

    #[test]
    fn test_g2_basis_dimension() {
        // Der(O) = g2 should have dimension 14
        let basis = compute_g2_basis();
        assert_eq!(
            basis.len(),
            14,
            "g2 basis should have 14 elements, got {}",
            basis.len()
        );
    }

    #[test]
    fn test_derivations_map_scalars_to_zero() {
        let basis = compute_g2_basis();
        for (idx, d) in basis.iter().enumerate() {
            assert!(
                d.maps_scalars_to_zero(1e-10),
                "Derivation {} must annihilate the identity",
                idx
            );
        }
    }

    #[test]
    fn test_derivations_leibniz_exhaustive() {
        // Every basis derivation must satisfy Leibniz on ALL basis pairs
        let basis = compute_g2_basis();
        let mut pass_count = 0;
        let mut total = 0;
        for (idx, d) in basis.iter().enumerate() {
            if d.frobenius_norm() < 1e-12 {
                continue;
            }
            for i in 0..8 {
                for j in 0..8 {
                    let ei = Octonion::basis(i);
                    let ej = Octonion::basis(j);
                    total += 1;
                    if d.verify_leibniz(&ei, &ej, 1e-8) {
                        pass_count += 1;
                    } else {
                        eprintln!("Leibniz failed: derivation {}, basis ({}, {})", idx, i, j);
                    }
                }
            }
        }
        assert_eq!(
            pass_count, total,
            "All derivations must satisfy Leibniz: {}/{} passed",
            pass_count, total
        );
    }

    #[test]
    fn test_derivations_antisymmetry() {
        // <D(x), y> + <x, D(y)> = 0 (skew-symmetric w.r.t. inner product)
        let basis = compute_g2_basis();
        for (idx, d) in basis.iter().enumerate() {
            if d.frobenius_norm() < 1e-12 {
                continue;
            }
            for i in 0..8 {
                for j in 0..8 {
                    let ei = Octonion::basis(i);
                    let ej = Octonion::basis(j);
                    assert!(
                        d.verify_antisymmetry(&ei, &ej, 1e-8),
                        "Derivation {} not anti-symmetric on ({}, {})",
                        idx,
                        i,
                        j
                    );
                }
            }
        }
    }

    #[test]
    fn test_three_form_alternating() {
        let e1 = Octonion::basis(1);
        let e2 = Octonion::basis(2);
        let e4 = Octonion::basis(4);
        assert!(G2Structure::verify_three_form_alternating(
            &e1, &e2, &e4, 1e-10
        ));

        let x = Octonion::new([0.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let y = Octonion::new([0.0, 0.0, 0.3, 0.7, 0.0, 0.0, 0.0, 0.0]);
        let z = Octonion::new([0.0, 0.0, 0.0, 0.0, 1.0, 0.2, 0.0, 0.0]);
        assert!(G2Structure::verify_three_form_alternating(
            &x, &y, &z, 1e-10
        ));
    }

    #[test]
    fn test_three_form_values() {
        let e1 = Octonion::basis(1);
        let e2 = Octonion::basis(2);
        let e3 = Octonion::basis(3);
        let phi = G2Structure::three_form(&e1, &e2, &e3);
        // e2*e3 = e1, so <e1, e1> = 1
        assert!((phi - 1.0).abs() < 1e-10, "phi(e1,e2,e3) = 1, got {}", phi);

        let e4 = Octonion::basis(4);
        let phi2 = G2Structure::three_form(&e1, &e2, &e4);
        // e2*e4 = e6, <e1, e6> = 0
        assert!(
            phi2.abs() < 1e-10,
            "phi(e1,e2,e4) should be 0, got {}",
            phi2
        );
    }

    #[test]
    fn test_inner_product_from_cross_product() {
        let a = Octonion::new([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let b = Octonion::new([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert!(
            G2Structure::verify_inner_product_from_cross(&a, &b, 1e-8),
            "Inner product must be reconstructible from cross product"
        );

        let c = Octonion::new([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert!(G2Structure::verify_inner_product_from_cross(&a, &c, 1e-8));
    }

    #[test]
    fn test_identity_is_automorphism() {
        let id = |x: &Octonion| -> Octonion { *x };
        let e1 = Octonion::basis(1);
        let e2 = Octonion::basis(2);
        assert!(G2Structure::verify_cross_product_preserved(
            &id, &e1, &e2, 1e-10
        ));
    }

    #[test]
    fn test_three_form_fully_alternating() {
        let e1 = Octonion::basis(1);
        let e2 = Octonion::basis(2);
        let e3 = Octonion::basis(3);

        let phi_123 = G2Structure::three_form(&e1, &e2, &e3);
        let phi_132 = G2Structure::three_form(&e1, &e3, &e2);
        let phi_213 = G2Structure::three_form(&e2, &e1, &e3);
        let phi_231 = G2Structure::three_form(&e2, &e3, &e1);
        let phi_312 = G2Structure::three_form(&e3, &e1, &e2);
        let phi_321 = G2Structure::three_form(&e3, &e2, &e1);

        // Even permutations: same sign
        assert!((phi_123 - phi_231).abs() < 1e-10);
        assert!((phi_123 - phi_312).abs() < 1e-10);

        // Odd permutations: opposite sign
        assert!((phi_123 + phi_132).abs() < 1e-10);
        assert!((phi_123 + phi_213).abs() < 1e-10);
        assert!((phi_123 + phi_321).abs() < 1e-10);
    }

    #[test]
    fn test_derivation_leibniz_random_octonions() {
        // Test Leibniz with non-basis octonions to ensure generality
        let basis = compute_g2_basis();
        let x = Octonion::new([1.0, 0.3, -0.5, 0.7, 0.1, -0.2, 0.4, 0.6]);
        let y = Octonion::new([0.5, -0.1, 0.8, 0.0, -0.3, 0.9, -0.4, 0.2]);

        for (idx, d) in basis.iter().enumerate() {
            assert!(
                d.verify_leibniz(&x, &y, 1e-8),
                "Leibniz failed for derivation {} on random octonions",
                idx
            );
        }
    }
}
