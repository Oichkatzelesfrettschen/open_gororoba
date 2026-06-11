//! Real-part projection for Cayley-Dickson algebra elements.
//!
//! In any Cayley-Dickson algebra of dimension n, the basis is {e_0, e_1, ..., e_{n-1}}
//! where e_0 = 1 is the real unit. The e_0 component is the unique direction that is
//! commutative and associative with all other elements.
//!
//! This module provides pure algebraic operations for extracting and analyzing
//! the real (e_0) component of CD algebra elements and operators.
//!
//! Basis ordering convention: elements are stored as `[e_0, e_1, ..., e_{n-1}]`
//! where index 0 is the real part.

/// Extract the real (e_0) component of a CD algebra element.
///
/// This is the projection onto the unique commutative-associative direction.
///
/// # Panics
/// Panics if `element` is empty.
pub fn project(element: &[f64]) -> f64 {
    assert!(
        !element.is_empty(),
        "element must have at least one component"
    );
    element[0]
}

/// Compute the scalar fraction: |e_0 component|^2 / |full element|^2.
///
/// Returns the fraction of the element's norm-squared that lives in the
/// real (e_0) direction. Bounded in `[0, 1]`.
///
/// Returns 0.0 if the element is the zero vector.
pub fn scalar_fraction(element: &[f64]) -> f64 {
    assert!(
        !element.is_empty(),
        "element must have at least one component"
    );
    let norm_sq: f64 = element.iter().map(|x| x * x).sum();
    if norm_sq == 0.0 {
        return 0.0;
    }
    let real_sq = element[0] * element[0];
    real_sq / norm_sq
}

/// Read the (0, 0) entry of a left-multiplication matrix.
///
/// Given a CD algebra element `a`, the left-multiplication operator L_a
/// is the linear map x -> a * x, representable as a dim x dim matrix.
/// The (0, 0) entry of L_a equals the real part of `a` (since e_0 * e_0 = e_0).
///
/// This function reads that entry from a flattened row-major matrix.
///
/// Basis ordering: e_0, e_1, ..., e_{dim-1} (index 0 = real part).
///
/// # Panics
/// Panics if `matrix.len() < dim * dim` or `dim == 0`.
pub fn operator_real_entry(matrix: &[f64], dim: usize) -> f64 {
    assert!(dim > 0, "dimension must be positive");
    assert!(
        matrix.len() >= dim * dim,
        "matrix length {} too small for dim {}",
        matrix.len(),
        dim
    );
    matrix[0] // (0, 0) entry in row-major layout
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_project_returns_real_component() {
        // Octonion (dim 8)
        let oct = [3.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert_eq!(project(&oct), 3.0);

        // Sedenion (dim 16)
        let mut sed = [0.0; 16];
        sed[0] = -2.5;
        sed[7] = 1.0;
        assert_eq!(project(&sed), -2.5);

        // Real (dim 1)
        assert_eq!(project(&[42.0]), 42.0);
    }

    #[test]
    fn test_scalar_fraction_bounds() {
        // Pure real: fraction = 1.0
        let real = [5.0, 0.0, 0.0, 0.0];
        assert!((scalar_fraction(&real) - 1.0).abs() < 1e-15);

        // Pure imaginary: fraction = 0.0
        let imag = [0.0, 1.0, 2.0, 3.0];
        assert!(scalar_fraction(&imag).abs() < 1e-15);

        // Mixed: 0 < fraction < 1
        let mixed = [1.0, 1.0, 1.0, 1.0];
        let frac = scalar_fraction(&mixed);
        assert!(frac > 0.0 && frac < 1.0);
        assert!((frac - 0.25).abs() < 1e-15); // 1/4

        // Zero vector
        let zero = [0.0, 0.0, 0.0, 0.0];
        assert_eq!(scalar_fraction(&zero), 0.0);
    }

    #[test]
    fn test_scalar_fraction_exact_on_basis() {
        // e_0 basis: fraction = 1.0
        let e0 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert!((scalar_fraction(&e0) - 1.0).abs() < 1e-15);

        // e_k basis (k > 0): fraction = 0.0
        for k in 1..8 {
            let mut ek = [0.0; 8];
            ek[k] = 1.0;
            assert!(
                scalar_fraction(&ek).abs() < 1e-15,
                "e_{} should have scalar fraction 0, got {}",
                k,
                scalar_fraction(&ek)
            );
        }
    }

    #[test]
    fn test_operator_real_entry_basis() {
        // For left-multiplication by e_0 = 1 (identity), the matrix is I.
        // So (0,0) entry = 1.0.
        let dim = 4;
        let mut identity = vec![0.0; dim * dim];
        for i in 0..dim {
            identity[i * dim + i] = 1.0;
        }
        assert!((operator_real_entry(&identity, dim) - 1.0).abs() < 1e-15);

        // For left-multiplication by e_1 (pure imaginary), (0,0) entry = 0.0
        // because e_1 * e_0 has no e_0 component.
        let mut l_e1 = vec![0.0; dim * dim];
        l_e1[1] = 1.0; // e_1 * e_0 -> e_1 (no real part)
        assert!(operator_real_entry(&l_e1, dim).abs() < 1e-15);
    }

    #[test]
    fn test_project_idempotent() {
        let element = [std::f64::consts::PI, 2.71, 1.41, 0.0];
        let p = project(&element);
        // project of a scalar is itself
        assert_eq!(project(&[p]), p);
    }
}
