//! S3-invariant bilinear form and democratic mixing (Rocq-verified: C-885).
//!
//! An S3-invariant bilinear form on a 3-element set has two free parameters:
//! diagonal value `a` and off-diagonal value `b`.  When `a = b` (uniform
//! mixing), the matrix is democratic with all entries equal.  For doubly
//! stochastic normalization (row sum = 1), this forces `a = b = 1/3`.
//! Kernel-checked in `proofs/verified/C885_DemocraticMixing.v`.

/// Check whether a 3x3 matrix is democratic (all entries equal within tolerance).
pub fn is_democratic(matrix: &[[f64; 3]; 3], tol: f64) -> bool {
    let val = matrix[0][0];
    matrix
        .iter()
        .all(|row| row.iter().all(|&x| (x - val).abs() < tol))
}

/// Check whether a 3x3 matrix is S3-invariant (diagonal entries equal,
/// off-diagonal entries equal).
pub fn is_s3_invariant(matrix: &[[f64; 3]; 3], tol: f64) -> bool {
    let diag = matrix[0][0];
    let offdiag = matrix[0][1];
    // Check all diagonal entries
    let diag_ok = (matrix[1][1] - diag).abs() < tol && (matrix[2][2] - diag).abs() < tol;
    // Check all off-diagonal entries
    let offdiag_ok = (matrix[0][2] - offdiag).abs() < tol
        && (matrix[1][0] - offdiag).abs() < tol
        && (matrix[1][2] - offdiag).abs() < tol
        && (matrix[2][0] - offdiag).abs() < tol
        && (matrix[2][1] - offdiag).abs() < tol;
    diag_ok && offdiag_ok
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn democratic_matrix() {
        let m = [[1.0 / 3.0; 3]; 3];
        assert!(is_democratic(&m, 1e-15));
        assert!(is_s3_invariant(&m, 1e-15));
    }

    #[test]
    fn s3_invariant_not_democratic() {
        // Diagonal a=0.5, off-diagonal b=0.25 -- S3-invariant but not democratic
        let m = [[0.5, 0.25, 0.25], [0.25, 0.5, 0.25], [0.25, 0.25, 0.5]];
        assert!(is_s3_invariant(&m, 1e-15));
        assert!(!is_democratic(&m, 1e-10));
    }

    #[test]
    fn non_s3_invariant() {
        let m = [[1.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.0, 0.5, 0.5]];
        assert!(!is_s3_invariant(&m, 1e-10));
    }

    #[test]
    fn democratic_normalization() {
        // If all entries equal a and row sum = 1, then 3a = 1 => a = 1/3
        let a = 1.0 / 3.0;
        assert!((a + a + a - 1.0_f64).abs() < 1e-15);
    }
}
