//! Division algebra criterion (Brown Theorem 3).
//!
//! A_4 = A_3{gamma} is a division algebra iff:
//! (a) A_3 is a division algebra
//! (b) gamma is NOT the norm of any element in A_3
//! (c) -gamma is NOT the norm of any trace-zero element in A_3
//!
//! Over R with gamma = -1:
//! (a) O is division: TRUE
//! (b) -1 is not a norm: TRUE (norms are non-negative)
//! (c) +1 is not a trace-zero norm: FALSE (n(e_1) = 1)
//! => Sedenions are NOT division over R.
//!
//! Mirrors: BrownGeneralizedCD.v (brown_condition_c_fails).


/// Check Brown's Theorem 3 conditions for A_4 = A_3{gamma} over R.
///
/// Returns (cond_a, cond_b, cond_c, is_division).
///
/// NOTE: Over R, norms are always non-negative (sum of squares).
/// So condition (b) holds iff gamma > 0 (negative gamma can't be a norm).
/// Condition (c) holds iff -gamma is not achievable as the norm of
/// any trace-zero (purely imaginary) element.
///
/// For gamma = -1: (b) holds (-1 < 0, not a norm), (c) fails (1 = n(e_1)).
/// For gamma = -2: (b) holds, (c) fails (2 = n(e_1 + e_2) since 1+1=2).
/// In fact, over R, (c) ALWAYS fails for gamma < 0 because -gamma > 0
/// and any positive real is achievable as the norm of a scaled basis element.
///
/// Mirrors: BrownGeneralizedCD.v brown_condition_c_fails.
pub fn check_division_criterion_over_r(
    base_dim: usize,
    gamma: f64,
) -> (bool, bool, bool, bool) {
    // (a) A_3 is division: true iff base_dim <= 8
    let cond_a = base_dim <= 8;

    // (b) gamma is not a norm: over R, norms >= 0, so true iff gamma < 0
    let cond_b = gamma < 0.0;

    // (c) -gamma is not a trace-zero norm: over R, trace-zero norms
    // cover all of [0, inf). So -gamma must be negative for (c) to hold.
    // -gamma < 0 iff gamma > 0.
    let cond_c = gamma > 0.0; // -gamma < 0, not achievable

    let is_division = cond_a && cond_b && cond_c;
    (cond_a, cond_b, cond_c, is_division)
}

/// Explain why the standard sedenions (gamma = -1) fail.
pub fn explain_standard_failure() -> &'static str {
    "Over R with gamma = -1:\n\
     (a) O is a division algebra: PASS\n\
     (b) gamma = -1 < 0, not a norm: PASS\n\
     (c) -gamma = +1 IS the norm of e_1 (trace-zero): FAIL\n\
     Sedenions are NOT a division algebra over R."
}

/// Check if a specific value is achievable as the norm of a
/// trace-zero element in the standard CD algebra at given dimension.
///
/// Over R, the norm of trace-zero elements ranges over [0, inf),
/// so ANY non-negative value is achievable.
pub fn is_trace_zero_norm_over_r(target: f64, dim: usize) -> bool {
    if target < 0.0 {
        return false;
    }
    if dim == 0 {
        return target == 0.0;
    }
    // Over R, any positive target is achievable: take x = sqrt(target) * e_1
    // which has norm = target and trace = 0.
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_sedenions_not_division() {
        let (a, b, c, div) = check_division_criterion_over_r(8, -1.0);
        assert!(a, "condition (a): O is division");
        assert!(b, "condition (b): -1 < 0");
        assert!(!c, "condition (c): +1 IS a trace-zero norm");
        assert!(!div, "sedenions should NOT be division");
    }

    #[test]
    fn test_no_gamma_works_over_r() {
        // Over R, conditions (b) and (c) are contradictory:
        // (b) requires gamma < 0, (c) requires gamma > 0.
        // So NO gamma value makes A_4 a division algebra over R.
        for gamma in [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0] {
            let (_, _, _, div) = check_division_criterion_over_r(8, gamma);
            assert!(!div, "gamma={gamma}: should NOT be division over R");
        }
    }

    #[test]
    fn test_quaternion_level_always_division() {
        // A_2 = quaternions: always division (base_dim = 2)
        // But Brown's criterion is for A_4, not A_2.
        // At base_dim = 4 (building dim 8 from dim 4):
        let (a, _, _, _) = check_division_criterion_over_r(4, -1.0);
        assert!(a, "H is division");
    }

    #[test]
    fn test_trace_zero_norm_coverage() {
        assert!(is_trace_zero_norm_over_r(1.0, 8));
        assert!(is_trace_zero_norm_over_r(2.0, 8));
        assert!(is_trace_zero_norm_over_r(0.0, 8));
        assert!(!is_trace_zero_norm_over_r(-1.0, 8));
    }

    #[test]
    fn test_explain() {
        let s = explain_standard_failure();
        assert!(s.contains("FAIL"));
        assert!(s.contains("NOT a division algebra"));
    }
}
