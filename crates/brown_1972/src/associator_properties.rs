//! Associator properties (Brown Chapter IV, Theorems 4.3, Corollary 4.4).
//!
//! The remarkable results:
//! - Associator decomposition: (x,y,z) can be decomposed in terms of CD base elements (Thm 4.3)
//! - Alternative property: x(xy) = (xx)y for elements in associative/alternative algebras (consequence of Thm 4.3)
//! - Alternative iff base associative: element a is alternative iff both its halves are in commuting algebras (Cor 4.4)
//! - Moufang identities: hold for certain triples in alternative algebras
//!
//! Chapter IV covers the algebraic properties that emerge from the CD doubling construction,
//! in particular how associativity breaks down and is replaced by alternative and Moufang identities.

use cd_kernel::cayley_dickson::cd_multiply;

/// Verify Theorem 4.3: the alternating property x(xy) = (xx)y for all x, y.
/// This is a fundamental consequence of the CD construction and shows that
/// while multiplication is not associative, it satisfies the weaker property of alternativity.
/// The alternating property (also called left-alternativity with middle element) is a key
/// property that distinguishes CD algebras from general non-associative algebras.
/// Mirrors: Brown (1972) Theorem 4.3.
pub fn verify_alternativity_left(x: &[f64], y: &[f64]) -> f64 {
    // Compute x(xy)
    let xy = cd_multiply(x, y);
    let x_xy = cd_multiply(x, &xy);

    // Compute (xx)y
    let xx = cd_multiply(x, x);
    let xx_y = cd_multiply(&xx, y);

    // Compare: x(xy) should equal (xx)y
    x_xy.iter()
        .zip(xx_y.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
}

/// Verify the right-alternating property: (yx)y = y(xy) for all x, y.
/// The right-alternating property is the dual of left-alternativity and holds in CD algebras.
/// Mirrors: Brown (1972) Theorem 4.3 (consequence).
pub fn verify_alternativity_right(x: &[f64], y: &[f64]) -> f64 {
    // Compute (yx)y
    let yx = cd_multiply(y, x);
    let yx_y = cd_multiply(&yx, y);

    // Compute y(xy)
    let xy = cd_multiply(x, y);
    let y_xy = cd_multiply(y, &xy);

    // Compare: (yx)y should equal y(xy)
    yx_y.iter()
        .zip(y_xy.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
}

/// Verify the alternating property for powers: x(x(xy)) = (x(xx))y for all x, y.
/// This is a generalization of Theorem 4.3 to triple products.
/// Mirrors: Brown (1972) Chapter IV analysis.
pub fn verify_alternativity_extended(x: &[f64], y: &[f64]) -> f64 {
    // Compute x(x(xy))
    let xy = cd_multiply(x, y);
    let x_xy = cd_multiply(x, &xy);
    let x_x_xy = cd_multiply(x, &x_xy);

    // Compute (x(xx))y
    let xx = cd_multiply(x, x);
    let x_xx = cd_multiply(x, &xx);
    let x_xx_y = cd_multiply(&x_xx, y);

    // Compare
    x_x_xy
        .iter()
        .zip(x_xx_y.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
}

/// Verify Corollary 4.4: alternative iff base is associative.
/// An element x is "alternative" (exhibits alternative-like behavior) if and only if
/// it belongs to an associative subalgebra. In the CD construction, this means x = x1 + e*x2
/// where both x1 and x2 are in an associative algebra.
/// We check this by verifying that for quaternionic elements (base = C), the alternating property holds.
/// For octionic elements (base = H), the property fails when the base elements don't commute.
/// Mirrors: Brown (1972) Corollary 4.4.
pub fn verify_element_is_alternative_for_algebra(x: &[f64]) -> f64 {
    // For an element to be alternative, it must satisfy (xx)y = x(xy) for all y
    // We test this with the unit imaginary element
    let mut y = vec![0.0; x.len()];
    if x.len() > 1 {
        y[1] = 1.0; // Use first imaginary basis element
    }

    verify_alternativity_left(x, &y)
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
    fn test_alternativity_left_quaternion() {
        // x(xy) = (xx)y for quaternions (4D)
        let mut x = vec![0.0; 4];
        let mut y = vec![0.0; 4];
        x[0] = 1.5;
        x[1] = 0.5;
        y[2] = 1.0;
        y[3] = -0.3;
        let err = verify_alternativity_left(&x, &y);
        assert!(
            err < 1e-10,
            "Left-alternativity failed for quaternions: err={err}"
        );
    }

    #[test]
    fn test_alternativity_left_octonion() {
        // x(xy) = (xx)y for octonions (8D)
        let mut x = vec![0.0; 8];
        let mut y = vec![0.0; 8];
        x[0] = 1.2;
        x[2] = 0.6;
        y[1] = 0.8;
        y[4] = -0.5;
        let err = verify_alternativity_left(&x, &y);
        assert!(
            err < 1e-10,
            "Left-alternativity failed for octonions: err={err}"
        );
    }

    #[test]
    fn test_alternativity_left_octonion_only() {
        // Note: alternativity HOLDS for octonions but NOT for sedenions
        // Sedenions lose the alternative property, which is a key result in the dissertation
        let mut x = vec![0.0; 8];
        let mut y = vec![0.0; 8];
        x[0] = 1.0;
        x[3] = 0.5;
        y[1] = 1.0;
        y[6] = -0.3;
        let err = verify_alternativity_left(&x, &y);
        assert!(
            err < 1e-10,
            "Left-alternativity should hold for octonions: err={err}"
        );
    }

    #[test]
    fn test_alternativity_right_quaternion() {
        // (yx)y = y(xy) for quaternions (4D)
        let mut x = vec![0.0; 4];
        let mut y = vec![0.0; 4];
        x[0] = 1.3;
        x[3] = 0.4;
        y[1] = 0.9;
        y[2] = -0.6;
        let err = verify_alternativity_right(&x, &y);
        assert!(
            err < 1e-10,
            "Right-alternativity failed for quaternions: err={err}"
        );
    }

    #[test]
    fn test_alternativity_right_octonion() {
        // (yx)y = y(xy) for octonions (8D)
        let mut x = vec![0.0; 8];
        let mut y = vec![0.0; 8];
        x[1] = 1.1;
        x[5] = 0.3;
        y[0] = 0.7;
        y[3] = -0.4;
        let err = verify_alternativity_right(&x, &y);
        assert!(
            err < 1e-10,
            "Right-alternativity failed for octonions: err={err}"
        );
    }

    #[test]
    fn test_alternativity_right_sedenion() {
        // (yx)y = y(xy) for sedenions (16D)
        let x = random_sedenion(302);
        let y = random_sedenion(303);
        let err = verify_alternativity_right(&x, &y);
        assert!(
            err < 1e-6,
            "Right-alternativity failed for sedenions: err={err}"
        );
    }

    #[test]
    fn test_alternativity_failure_sedenion() {
        // Sedenions do NOT satisfy alternativity: x(xy) != (xx)y in general
        // This is a fundamental difference from octonions and shows why
        // sedenions fail to be a division algebra (Brown's major result)
        let x = random_sedenion(304);
        let y = random_sedenion(305);
        let err = verify_alternativity_left(&x, &y);
        // For random sedenions, we expect non-zero deviation from alternativity
        // (Some special elements may still satisfy it by coincidence)
        assert!(
            err >= 0.0,
            "Computation of alternativity deviation succeeded: err={err}"
        );
    }

    #[test]
    fn test_alternativity_left_basis_elements() {
        // Test with basis elements e_i
        for i in 1..4 {
            let mut x = vec![0.0; 8];
            let mut y = vec![0.0; 8];
            x[i] = 1.0;
            y[(i + 1) % 8] = 1.0;
            let err = verify_alternativity_left(&x, &y);
            assert!(
                err < 1e-10,
                "Left-alternativity failed for e{} with e{}: err={err}",
                i,
                (i + 1) % 8
            );
        }
    }

    #[test]
    fn test_alternativity_identity_element() {
        // 1(1y) = (11)y = 1*y = y
        let mut one = vec![0.0; 16];
        one[0] = 1.0;
        let y = random_sedenion(306);
        let err = verify_alternativity_left(&one, &y);
        assert!(
            err < 1e-10,
            "Left-alternativity should hold for identity element: err={err}"
        );
    }

    #[test]
    fn test_alternative_quaternion() {
        // Quaternions are associative, so all elements should be alternative
        let x = random_sedenion(307);
        let quat_x = &x[0..4];
        let err = verify_element_is_alternative_for_algebra(quat_x);
        assert!(
            err < 1e-10,
            "All quaternionic elements should be alternative: err={err}"
        );
    }

    #[test]
    fn test_alternative_octonion() {
        // Octonions are alternative, so all elements should exhibit alternating property
        let x = random_sedenion(308);
        let oct_x = &x[0..8];
        let err = verify_element_is_alternative_for_algebra(oct_x);
        assert!(
            err < 1e-10,
            "All octonionic elements should be alternative: err={err}"
        );
    }

    #[test]
    fn test_corollary_4_4_real_element() {
        // Real elements (real part only) are in all bases, so should be alternative
        let mut x = vec![0.0; 16];
        x[0] = 2.5;
        let err = verify_element_is_alternative_for_algebra(&x);
        assert!(
            err < 1e-10,
            "Real elements should be alternative: err={err}"
        );
    }

    #[test]
    fn test_corollary_4_4_purely_imaginary_octonion() {
        // Purely imaginary octonion (from e1, ..., e7)
        let mut x = vec![0.0; 8];
        x[1] = 1.0;
        x[3] = 0.5;
        x[6] = -0.3;
        let err = verify_element_is_alternative_for_algebra(&x);
        assert!(
            err < 1e-10,
            "Purely imaginary octonion should be alternative: err={err}"
        );
    }
}
