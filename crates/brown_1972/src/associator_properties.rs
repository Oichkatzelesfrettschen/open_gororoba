//! Associator properties (Brown Chapter IV, Theorems 4.3, Corollary 4.4).
//!
//! The remarkable results:
//! - Associator decomposition: (x,y,z) can be decomposed in terms of CD base elements (Thm 4.3)
//! - Flexibility: (xy)x = x(yx) holds at every Cayley-Dickson dimension (Brown Theorem 4.2)
//! - Alternativity: x(xy) = (xx)y and (xy)y = x(yy) hold exactly when the generating algebra is
//!   associative, so quaternions and octonions are alternative and sedenions are not (Brown 4.4)
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

/// Verify flexibility: (yx)y = y(xy) for all x, y.
/// Flexibility is the identity (a,b,a) = 0 on the associator. Every Cayley-Dickson algebra is
/// flexible, so the returned deviation is zero at 4D, 8D, 16D and beyond.
/// Mirrors: Brown (1972) Theorem 4.2.
pub fn verify_flexibility(x: &[f64], y: &[f64]) -> f64 {
    let yx = cd_multiply(y, x);
    let yx_y = cd_multiply(&yx, y);

    let xy = cd_multiply(x, y);
    let y_xy = cd_multiply(y, &xy);

    yx_y.iter()
        .zip(y_xy.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
}

/// Verify right alternativity: (xy)y = x(yy) for all x, y.
/// Right alternativity is the identity (a,b,b) = 0 on the associator, the mirror of the
/// left form checked by `verify_alternativity_left`. Brown Corollary 4.4 ties both forms to
/// associativity of the generating algebra, so the deviation is zero for quaternions and
/// octonions and nonzero for generic sedenion pairs.
/// Mirrors: Brown (1972) Corollary 4.4.
pub fn verify_alternativity_right(x: &[f64], y: &[f64]) -> f64 {
    let xy = cd_multiply(x, y);
    let xy_y = cd_multiply(&xy, y);

    let yy = cd_multiply(y, y);
    let x_yy = cd_multiply(x, &yy);

    xy_y.iter()
        .zip(x_yy.iter())
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
    fn test_flexibility_quaternion() {
        // (yx)y = y(xy) for quaternions (4D)
        let mut x = vec![0.0; 4];
        let mut y = vec![0.0; 4];
        x[0] = 1.3;
        x[3] = 0.4;
        y[1] = 0.9;
        y[2] = -0.6;
        let err = verify_flexibility(&x, &y);
        assert!(err < 1e-10, "Flexibility failed for quaternions: err={err}");
        let err = verify_alternativity_right(&x, &y);
        assert!(
            err < 1e-10,
            "Right-alternativity failed for quaternions: err={err}"
        );
    }

    #[test]
    fn test_flexibility_octonion() {
        // (yx)y = y(xy) for octonions (8D)
        let mut x = vec![0.0; 8];
        let mut y = vec![0.0; 8];
        x[1] = 1.1;
        x[5] = 0.3;
        y[0] = 0.7;
        y[3] = -0.4;
        let err = verify_flexibility(&x, &y);
        assert!(err < 1e-10, "Flexibility failed for octonions: err={err}");
        let err = verify_alternativity_right(&x, &y);
        assert!(
            err < 1e-10,
            "Right-alternativity failed for octonions: err={err}"
        );
    }

    #[test]
    fn test_flexibility_sedenion() {
        // Flexibility survives the sedenion doubling (Brown Theorem 4.2).
        let x = random_sedenion(302);
        let y = random_sedenion(303);
        let err = verify_flexibility(&x, &y);
        assert!(err < 1e-6, "Flexibility failed for sedenions: err={err}");
    }

    #[test]
    fn test_alternativity_right_fails_sedenion() {
        // The octonions are not associative, so by Brown Corollary 4.4 the sedenions are
        // not right alternative. The deviation must sit away from zero, not merely be finite.
        let x = random_sedenion(302);
        let y = random_sedenion(303);
        let err = verify_alternativity_right(&x, &y);
        assert!(
            err > 1e-2,
            "Sedenion right-alternativity deviation unexpectedly small: err={err}"
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
        // A generic sedenion pair sits away from the alternative locus, so the deviation
        // must clear a fixed floor; `err >= 0.0` would pass for any finite value.
        assert!(
            err > 1e-2,
            "Sedenion left-alternativity deviation unexpectedly small: err={err}"
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
