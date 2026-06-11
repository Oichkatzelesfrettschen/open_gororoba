//! Inline CD operations for small dimensions via SmallVec.
//!
//! For d <= 16 (quaternion through sedenion), all CD multiplication
//! intermediates fit inline in SmallVec<`[f64; 16]`> without heap allocation.
//! This eliminates the Vec allocation overhead in the associator hot loop.
//!
//! Performance impact: at d=16, the associator triplet computation
//! requires 4 CD multiplications, each producing a 16-element result.
//! With SmallVec, all 4 results stay on the stack (64 * 8 = 512 bytes).

use smallvec::SmallVec;

/// Inline CD conjugation: negate all components except the first.
fn cd_conjugate_inline(a: &[f64]) -> SmallVec<[f64; 16]> {
    let mut result: SmallVec<[f64; 16]> = SmallVec::from_slice(a);
    for x in result[1..].iter_mut() {
        *x = -*x;
    }
    result
}

/// Inline CD multiplication for d <= 16.
///
/// Returns SmallVec that stays on the stack for d <= 16,
/// spills to heap only for larger dimensions.
pub fn cd_multiply_inline(a: &[f64], b: &[f64]) -> SmallVec<[f64; 16]> {
    let dim = a.len();
    debug_assert_eq!(a.len(), b.len());

    if dim == 1 {
        let mut result = SmallVec::new();
        result.push(a[0] * b[0]);
        return result;
    }

    let half = dim / 2;
    let (a_l, a_r) = (&a[..half], &a[half..]);
    let (c_l, c_r) = (&b[..half], &b[half..]);

    // Matching arith.rs cd_multiply exactly:
    // Lower: a_l * c_l - conj(c_r) * a_r
    // Upper: c_r * a_l + a_r * conj(c_l)
    let conj_c_r = cd_conjugate_inline(c_r);
    let term1 = cd_multiply_inline(a_l, c_l); // a_l * c_l
    let term2 = cd_multiply_inline(&conj_c_r, a_r); // conj(c_r) * a_r

    let conj_c_l = cd_conjugate_inline(c_l);
    let term3 = cd_multiply_inline(c_r, a_l); // c_r * a_l
    let term4 = cd_multiply_inline(a_r, &conj_c_l); // a_r * conj(c_l)

    let mut result = SmallVec::with_capacity(dim);
    for i in 0..half {
        result.push(term1[i] - term2[i]);
    }
    for i in 0..half {
        result.push(term3[i] + term4[i]);
    }
    result
}

/// Inline CD associator norm for d <= 16.
///
/// Computes ||`[a, b, c]`|| = ||(a*b)*c - a*(b*c)||
/// with zero heap allocation for d <= 16.
pub fn cd_associator_norm_inline(a: &[f64], b: &[f64], c: &[f64]) -> f64 {
    let ab = cd_multiply_inline(a, b);
    let abc_left = cd_multiply_inline(&ab, c);
    let bc = cd_multiply_inline(b, c);
    let abc_right = cd_multiply_inline(a, &bc);

    let mut norm_sq = 0.0f64;
    for i in 0..a.len() {
        let diff = abc_left[i] - abc_right[i];
        norm_sq += diff * diff;
    }
    norm_sq.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cd_associator_norm;

    #[test]
    fn test_inline_multiply_matches_standard() {
        // Quaternion (4D) test
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];

        let inline_result = cd_multiply_inline(&a, &b);
        let standard_result = crate::cayley_dickson::cd_multiply(&a, &b);

        for (i, (&inl, &std)) in inline_result.iter().zip(standard_result.iter()).enumerate() {
            assert!(
                (inl - std).abs() < 1e-10,
                "Quaternion multiply mismatch at {}: inline={}, standard={}",
                i,
                inl,
                std
            );
        }
    }

    #[test]
    fn test_inline_multiply_sedenion() {
        // Sedenion (16D) -- should still be inline (no heap)
        let a: Vec<f64> = (0..16).map(|i| (i as f64 * 0.1).sin()).collect();
        let b: Vec<f64> = (0..16).map(|i| (i as f64 * 0.2).cos()).collect();

        let inline_result = cd_multiply_inline(&a, &b);
        let standard_result = crate::cayley_dickson::cd_multiply(&a, &b);

        assert_eq!(inline_result.len(), 16);
        assert!(
            !inline_result.spilled(),
            "Sedenion should be inline (no heap)"
        );

        for (i, (&inl, &std)) in inline_result.iter().zip(standard_result.iter()).enumerate() {
            assert!(
                (inl - std).abs() < 1e-10,
                "Sedenion multiply mismatch at {}: inline={}, standard={}",
                i,
                inl,
                std
            );
        }
    }

    #[test]
    fn test_inline_associator_matches_standard() {
        // Sedenion (16D) -- non-trivial associator
        let a: Vec<f64> = (0..16).map(|i| (i as f64 * 0.3).sin()).collect();
        let b: Vec<f64> = (0..16).map(|i| (i as f64 * 0.5).cos()).collect();
        let c: Vec<f64> = (0..16).map(|i| (i as f64 * 0.7 + 1.0).sin()).collect();

        let inline_norm = cd_associator_norm_inline(&a, &b, &c);
        let standard_norm = cd_associator_norm(&a, &b, &c);

        assert!(
            (inline_norm - standard_norm).abs() < 1e-10,
            "Associator norm mismatch: inline={}, standard={}",
            inline_norm,
            standard_norm
        );
        assert!(inline_norm > 0.0, "Sedenion associator should be nonzero");
    }

    #[test]
    fn test_inline_octonion_nonzero_associator() {
        // Octonion (8D) -- first dimension with nonzero associator
        let a: Vec<f64> = (0..8).map(|i| (i as f64 * 0.2).sin()).collect();
        let b: Vec<f64> = (0..8).map(|i| (i as f64 * 0.4).cos()).collect();
        let c: Vec<f64> = (0..8).map(|i| (i as f64 * 0.6 + 0.5).sin()).collect();

        let norm = cd_associator_norm_inline(&a, &b, &c);
        assert!(norm > 0.0, "Octonion associator should be nonzero");
    }

    #[test]
    fn test_quaternion_associator_zero() {
        // Quaternion (4D) -- associative, so associator should be zero
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        let c = [9.0, 10.0, 11.0, 12.0];

        let norm = cd_associator_norm_inline(&a, &b, &c);
        assert!(
            norm < 1e-10,
            "Quaternion should be associative, got norm={}",
            norm
        );
    }
}
