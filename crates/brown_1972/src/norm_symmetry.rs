//! Norm symmetry properties (Brown Chapter III, Theorems 3.9, Lemma 3.10).
//!
//! The remarkable results:
//! - Conjugate right operand: N(xy~) = N(xy) (Thm 3.9 part i)
//! - Conjugate left operand: N(x~y) = N(xy) (Thm 3.9 part ii)
//! - Norm symmetry: N(xy) = N(yx) (Thm 3.9 part iii)
//! - Polarization identity: N(x+y) + N(x-y) = 2(N(x) + N(y)) (Lemma 3.10)
//!
//! These hold in any flexible algebra with involution (CD algebras are flexible).
//! The proof relies on trace being in the center of the algebra.

use cd_kernel::cayley_dickson::{cd_conjugate, cd_multiply, cd_norm_sq};

/// Verify Theorem 3.9 part i: N(xy~) = N(xy).
/// The norm of a product with right operand conjugated equals the norm of the original product.
/// Mirrors: Brown (1972) Theorem 3.9 part i.
pub fn verify_norm_conjugate_right(x: &[f64], y: &[f64]) -> f64 {
    let xy = cd_multiply(x, y);
    let norm_xy = cd_norm_sq(&xy);

    let y_conj = cd_conjugate(y);
    let xy_conj = cd_multiply(x, &y_conj);
    let norm_xy_conj = cd_norm_sq(&xy_conj);

    (norm_xy - norm_xy_conj).abs()
}

/// Verify Theorem 3.9 part ii: N(x~y) = N(xy).
/// The norm of a product with left operand conjugated equals the norm of the original product.
/// Mirrors: Brown (1972) Theorem 3.9 part ii.
pub fn verify_norm_conjugate_left(x: &[f64], y: &[f64]) -> f64 {
    let xy = cd_multiply(x, y);
    let norm_xy = cd_norm_sq(&xy);

    let x_conj = cd_conjugate(x);
    let conj_x_y = cd_multiply(&x_conj, y);
    let norm_conj_x_y = cd_norm_sq(&conj_x_y);

    (norm_xy - norm_conj_x_y).abs()
}

/// Verify Theorem 3.9 part iii: N(xy) = N(yx).
/// Norm is symmetric under multiplication order swap.
/// This is the KEY property: norm(xy) = norm(yx) for all x, y in CD algebras.
/// Mirrors: Brown (1972) Theorem 3.9 part iii.
pub fn verify_norm_symmetry(x: &[f64], y: &[f64]) -> f64 {
    let xy = cd_multiply(x, y);
    let norm_xy = cd_norm_sq(&xy);

    let yx = cd_multiply(y, x);
    let norm_yx = cd_norm_sq(&yx);

    (norm_xy - norm_yx).abs()
}

/// Verify Lemma 3.10: the polarization identity.
/// For any x, y in a CD algebra: N(x+y) + N(x-y) = 2(N(x) + N(y)).
/// This fundamental identity relates norms of sums and differences and underlies
/// the existence of a bilinear form dual to the quadratic form N(x).
/// The identity holds in any algebra with a non-degenerate quadratic form.
/// Mirrors: Brown (1972) Lemma 3.10.
pub fn verify_polarization_identity(x: &[f64], y: &[f64]) -> f64 {
    // Compute N(x) and N(y)
    let norm_x = cd_norm_sq(x);
    let norm_y = cd_norm_sq(y);

    // Compute x + y and N(x + y)
    let x_plus_y: Vec<f64> = x.iter().zip(y.iter()).map(|(a, b)| a + b).collect();
    let norm_x_plus_y = cd_norm_sq(&x_plus_y);

    // Compute x - y and N(x - y)
    let x_minus_y: Vec<f64> = x.iter().zip(y.iter()).map(|(a, b)| a - b).collect();
    let norm_x_minus_y = cd_norm_sq(&x_minus_y);

    // Check: N(x+y) + N(x-y) = 2(N(x) + N(y))
    let lhs = norm_x_plus_y + norm_x_minus_y;
    let rhs = 2.0 * (norm_x + norm_y);

    (lhs - rhs).abs()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_sedenion(seed: u64) -> Vec<f64> {
        let mut x = vec![0.0; 16];
        let mut s = seed;
        for v in &mut x {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *v = ((s >> 33) as f64) / (1u64 << 31) as f64 - 1.0;
        }
        x
    }

    #[test]
    fn test_norm_conjugate_right_dim16() {
        let x = random_sedenion(11);
        let y = random_sedenion(12);
        let err = verify_norm_conjugate_right(&x, &y);
        assert!(
            err < 1e-10,
            "N(xy~) should equal N(xy) for sedenions: err={err}"
        );
    }

    #[test]
    fn test_norm_conjugate_left_dim16() {
        let x = random_sedenion(13);
        let y = random_sedenion(14);
        let err = verify_norm_conjugate_left(&x, &y);
        assert!(
            err < 1e-10,
            "N(x~y) should equal N(xy) for sedenions: err={err}"
        );
    }

    #[test]
    fn test_norm_symmetry_dim16() {
        let x = random_sedenion(15);
        let y = random_sedenion(16);
        let err = verify_norm_symmetry(&x, &y);
        assert!(
            err < 1e-10,
            "N(xy) should equal N(yx) for sedenions: err={err}"
        );
    }

    #[test]
    fn test_norm_symmetry_octonion() {
        // Test with octonions (8D)
        let mut x = vec![0.0; 8];
        let mut y = vec![0.0; 8];
        x[1] = 1.0;
        x[3] = 0.5;
        y[2] = 1.0;
        y[5] = -0.3;
        let err = verify_norm_symmetry(&x, &y);
        assert!(
            err < 1e-10,
            "N(xy) should equal N(yx) for octonions: err={err}"
        );
    }

    #[test]
    fn test_norm_symmetry_quaternion() {
        // Test with quaternions (4D)
        let mut x = vec![0.0; 4];
        let mut y = vec![0.0; 4];
        x[0] = 1.0;
        x[2] = 0.5;
        y[1] = 1.0;
        y[3] = -0.7;
        let err = verify_norm_symmetry(&x, &y);
        assert!(
            err < 1e-10,
            "N(xy) should equal N(yx) for quaternions: err={err}"
        );
    }

    #[test]
    fn test_all_three_parts_together() {
        // Verify all three parts of Theorem 3.9 simultaneously
        let x = random_sedenion(100);
        let y = random_sedenion(101);

        let err_conj_right = verify_norm_conjugate_right(&x, &y);
        let err_conj_left = verify_norm_conjugate_left(&x, &y);
        let err_symm = verify_norm_symmetry(&x, &y);

        assert!(err_conj_right < 1e-10, "Part i failed: N(xy~) != N(xy)");
        assert!(err_conj_left < 1e-10, "Part ii failed: N(x~y) != N(xy)");
        assert!(err_symm < 1e-10, "Part iii failed: N(xy) != N(yx)");
    }

    #[test]
    fn test_polarization_identity_quaternion() {
        // Test polarization identity in quaternions (4D)
        let mut x = vec![0.0; 4];
        let mut y = vec![0.0; 4];
        x[0] = 1.5;
        x[1] = 0.5;
        y[2] = 1.0;
        y[3] = -0.3;
        let err = verify_polarization_identity(&x, &y);
        assert!(
            err < 1e-10,
            "Polarization identity failed for quaternions: err={err}"
        );
    }

    #[test]
    fn test_polarization_identity_octonion() {
        // Test polarization identity in octonions (8D)
        let mut x = vec![0.0; 8];
        let mut y = vec![0.0; 8];
        x[0] = 1.2;
        x[2] = 0.6;
        x[4] = -0.4;
        y[1] = 0.8;
        y[3] = -0.5;
        y[6] = 0.3;
        let err = verify_polarization_identity(&x, &y);
        assert!(
            err < 1e-10,
            "Polarization identity failed for octonions: err={err}"
        );
    }

    #[test]
    fn test_polarization_identity_sedenion() {
        // Test polarization identity in sedenions (16D)
        let x = random_sedenion(200);
        let y = random_sedenion(201);
        let err = verify_polarization_identity(&x, &y);
        assert!(
            err < 1e-6,
            "Polarization identity failed for sedenions: err={err}"
        );
    }

    #[test]
    fn test_polarization_identity_zero() {
        // When one element is zero: N(x) + N(-x) = 2N(x)
        let x = random_sedenion(202);
        let zero = vec![0.0; 16];
        let err = verify_polarization_identity(&x, &zero);
        assert!(
            err < 1e-10,
            "Polarization identity should hold when y=0: err={err}"
        );
    }

    #[test]
    fn test_polarization_identity_same_element() {
        // When x = y: N(2x) + N(0) = 2(N(x) + N(x))
        // Which is: 4*N(x) = 4*N(x) <EMOJI+2713>
        let x = random_sedenion(203);
        let err = verify_polarization_identity(&x, &x);
        assert!(
            err < 1e-10,
            "Polarization identity should hold when y=x: err={err}"
        );
    }
}
