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
//! - G2 preserves the cross product on Im(O): x * y = [x,y]/2
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
    // Step 1: Parameterize a general element of so(7) acting on Im(O).
    // A 7x7 skew-symmetric matrix has 21 independent entries: M_{ij} for 1 <= i < j <= 7.
    // We index them as parameters p_0, p_1, ..., p_20.
    // Parameter k corresponds to entry (i,j) where i < j, enumerated row-by-row.

    // Map parameter index to (i,j) pair (0-indexed within Im(O), so 0..6)
    let mut param_pairs: Vec<(usize, usize)> = Vec::new();
    for i in 0..7 {
        for j in (i + 1)..7 {
            param_pairs.push((i, j));
        }
    }
    assert_eq!(param_pairs.len(), 21);

    // Step 2: Build the Leibniz constraint matrix.
    // For each pair (i,j) with 1 <= i <= j <= 7 (basis indices in O, so i,j in 1..=7):
    //   D(e_i * e_j) = D(e_i) * e_j + e_i * D(e_j)
    // This gives 8 equations per pair. We collect constraints as rows of A*p = 0.

    // For each of the 21 parameters, construct the 8x8 matrix with that parameter = 1.
    let mut param_matrices: Vec<[[f64; 8]; 8]> = Vec::new();
    for &(a, b) in &param_pairs {
        let mut m = [[0.0f64; 8]; 8];
        // Indices in O-space are a+1, b+1 (offset by 1 for scalar slot)
        m[a + 1][b + 1] = 1.0;
        m[b + 1][a + 1] = -1.0; // skew-symmetric
        param_matrices.push(m);
    }

    // Collect constraint equations
    let mut constraint_rows: Vec<[f64; 21]> = Vec::new();

    for i in 1..8 {
        for j in i..8 {
            let ei = Octonion::basis(i);
            let ej = Octonion::basis(j);
            let ei_ej = ei.multiply(&ej);

            // For each output component c (0..8):
            // sum_k p_k * [ M_k applied to (e_i*e_j) ]_c
            //   = sum_k p_k * [ (M_k e_i)*e_j + e_i*(M_k e_j) ]_c
            for c in 0..8 {
                let mut row = [0.0; 21];
                for (k, pm) in param_matrices.iter().enumerate() {
                    // LHS: D(e_i * e_j) component c
                    let mut lhs_c = 0.0;
                    #[allow(clippy::needless_range_loop)]
                    for s in 0..8 {
                        lhs_c += pm[c][s] * ei_ej.components[s];
                    }

                    // RHS: D(e_i)*e_j + e_i*D(e_j) component c
                    // D(e_i) = M_k * e_i
                    let mut d_ei = [0.0; 8];
                    #[allow(clippy::needless_range_loop)]
                    for s in 0..8 {
                        d_ei[s] = pm[s][i];
                    }
                    let d_ei_oct = Octonion::new(d_ei);
                    let d_ei_ej = d_ei_oct.multiply(&ej);

                    let mut d_ej = [0.0; 8];
                    for s in 0..8 {
                        d_ej[s] = pm[s][j];
                    }
                    let d_ej_oct = Octonion::new(d_ej);
                    let ei_d_ej = ei.multiply(&d_ej_oct);

                    let rhs_c = d_ei_ej.components[c] + ei_d_ej.components[c];

                    row[k] = lhs_c - rhs_c;
                }

                // Only add non-trivial constraints
                let norm: f64 = row.iter().map(|x| x * x).sum::<f64>().sqrt();
                if norm > 1e-14 {
                    constraint_rows.push(row);
                }
            }
        }
    }

    // Step 3: Find the null space of the constraint matrix via Gram-Schmidt.
    // First, find a basis for the row space, then the null space is the complement.

    // Reduce the constraint matrix to row echelon form via Gaussian elimination.
    let n_params = 21;
    let n_constraints = constraint_rows.len();
    let mut matrix = vec![vec![0.0f64; n_params]; n_constraints];
    for (i, row) in constraint_rows.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            matrix[i][j] = val;
        }
    }

    // Gaussian elimination with partial pivoting
    let mut pivot_cols: Vec<usize> = Vec::new();
    let mut current_row = 0;
    for col in 0..n_params {
        // Find pivot
        let mut max_val = 0.0;
        let mut max_row = current_row;
        #[allow(clippy::needless_range_loop)]
        for row in current_row..n_constraints {
            if matrix[row][col].abs() > max_val {
                max_val = matrix[row][col].abs();
                max_row = row;
            }
        }
        if max_val < 1e-12 {
            continue; // Free variable
        }

        // Swap rows
        matrix.swap(current_row, max_row);
        pivot_cols.push(col);

        // Eliminate below
        let pivot = matrix[current_row][col];
        for row in 0..n_constraints {
            if row == current_row {
                continue;
            }
            let factor = matrix[row][col] / pivot;
            #[allow(clippy::needless_range_loop)]
            for c in 0..n_params {
                matrix[row][c] -= factor * matrix[current_row][c];
            }
        }
        current_row += 1;
    }

    let rank = pivot_cols.len();
    let null_dim = n_params - rank;

    // Step 4: Extract null space vectors (free variable approach).
    // For each free variable, set it to 1 and solve for pivot variables.
    let free_cols: Vec<usize> = (0..n_params).filter(|c| !pivot_cols.contains(c)).collect();

    let mut null_basis: Vec<[f64; 21]> = Vec::new();
    for &free_col in &free_cols {
        let mut vec = [0.0; 21];
        vec[free_col] = 1.0;

        // Back-substitute to find pivot variable values
        for (pivot_idx, &pivot_col) in pivot_cols.iter().enumerate().rev() {
            let pivot_val = matrix[pivot_idx][pivot_col];
            if pivot_val.abs() < 1e-14 {
                continue;
            }
            let mut sum = 0.0;
            for c in 0..n_params {
                if c != pivot_col {
                    sum += matrix[pivot_idx][c] * vec[c];
                }
            }
            vec[pivot_col] = -sum / pivot_val;
        }
        null_basis.push(vec);
    }

    // Step 5: Convert null space vectors to OctonionDerivation objects.
    let mut derivations: Vec<OctonionDerivation> = Vec::new();
    for null_vec in &null_basis {
        let mut m = [[0.0f64; 8]; 8];
        for (k, &coeff) in null_vec.iter().enumerate() {
            let (a, b) = param_pairs[k];
            m[a + 1][b + 1] += coeff;
            m[b + 1][a + 1] -= coeff;
        }
        derivations.push(OctonionDerivation { matrix: m });
    }

    // Verify we got the expected dimension
    assert_eq!(
        null_dim, 14,
        "Der(O) should have dimension 14, got {} (rank={}, constraints={})",
        null_dim, rank, n_constraints
    );

    derivations
}

/// G2 structure computations on the imaginary octonions Im(O) = R^7.
///
/// G2 is characterized as the subgroup of GL(7,R) preserving:
/// 1. The inner product on Im(O)
/// 2. The cross product: x * y = [x,y]/2
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
