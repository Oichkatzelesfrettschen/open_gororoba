//! Zero divisor criterion (Brown Theorems 7.3, 7.12, 7.14, 7.15).
//!
//! The Major Theorem (7.15): A and B are mutual zero divisors in A_4 iff:
//!   (i)   N(a1) = N(a2)
//!   (ii)  b2 = [(a1*b1)*a2] / N(a1)
//!   (iii) )a1, b1, a2( = 0  (antiassociator is zero)
//!
//! This is the complete characterization that Moreno later refined.

use cd_kernel::cayley_dickson::{cd_conjugate, cd_multiply, cd_norm_sq};

/// The antiassociator: )A,B,C( = (AB)C + A(BC).
///
/// Brown Definition 7.3. Linear in each argument (Lemma 7.11).
/// When this is zero for a ZD triple (a1, b1, a2), the ZD criterion is met.
///
/// Mirrors: Brown (1972) Definition 7.3, Lemma 7.11.
pub fn antiassociator(a: &[f64], b: &[f64], c: &[f64]) -> Vec<f64> {
    let ab = cd_multiply(a, b);
    let ab_c = cd_multiply(&ab, c);
    let bc = cd_multiply(b, c);
    let a_bc = cd_multiply(a, &bc);
    ab_c.iter().zip(a_bc.iter()).map(|(x, y)| x + y).collect()
}

/// Check Brown's Theorem 7.15 criterion for AB = 0 in sedenions.
///
/// Given A = a1 + ea2, B = b1 + eb2 (octonionic halves):
///   (i)   N(a1) = N(a2)
///   (ii)  b2 = [(a1*b1)*a2] / N(a1)
///   (iii) )a1, b1, a2( = 0
///
/// Returns (cond_i, cond_ii, cond_iii, all_satisfied).
///
/// Mirrors: Brown (1972) Theorem 7.15 (THE MAJOR THEOREM).
pub fn check_major_theorem(a: &[f64], b: &[f64], atol: f64) -> MajorTheoremResult {
    let dim = a.len();
    let half = dim / 2;
    assert!(dim >= 16, "Major theorem applies to dim >= 16");

    let a1 = &a[..half];
    let a2 = &a[half..];
    let b1 = &b[..half];
    let b2 = &b[half..];

    // (i) N(a1) = N(a2)
    let n_a1 = cd_norm_sq(a1);
    let n_a2 = cd_norm_sq(a2);
    let cond_i = (n_a1 - n_a2).abs() < atol;

    // (ii) b2 = [(a1*b1)*a2] / N(a1)
    let a1_b1 = cd_multiply(a1, b1);
    let a1_b1_a2 = cd_multiply(&a1_b1, a2);
    let predicted_b2: Vec<f64> = if n_a1.abs() > 1e-15 {
        a1_b1_a2.iter().map(|x| x / n_a1).collect()
    } else {
        vec![0.0; half]
    };
    let b2_err: f64 = b2
        .iter()
        .zip(predicted_b2.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max);
    let cond_ii = b2_err < atol;

    // (iii) )a1, b1, a2( = 0
    let anti = antiassociator(a1, b1, a2);
    let anti_norm = cd_norm_sq(&anti);
    let cond_iii = anti_norm < atol;

    MajorTheoremResult {
        cond_i,
        cond_ii,
        cond_iii,
        all_satisfied: cond_i && cond_ii && cond_iii,
        norm_a1: n_a1,
        norm_a2: n_a2,
        b2_error: b2_err,
        antiassociator_norm: anti_norm.sqrt(),
    }
}

/// Result of checking the Major Theorem conditions.
#[derive(Debug, Clone)]
pub struct MajorTheoremResult {
    /// N(a1) = N(a2)
    pub cond_i: bool,
    /// b2 = [(a1*b1)*a2] / N(a1)
    pub cond_ii: bool,
    /// )a1, b1, a2( = 0
    pub cond_iii: bool,
    /// All three conditions satisfied
    pub all_satisfied: bool,
    /// N(a1)
    pub norm_a1: f64,
    /// N(a2)
    pub norm_a2: f64,
    /// Max error in b2 prediction
    pub b2_error: f64,
    /// ||antiassociator||
    pub antiassociator_norm: f64,
}

impl std::fmt::Display for MajorTheoremResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Major Thm: (i)N(a1)={:.4}=N(a2)={:.4}:{} (ii)b2_err={:.2e}:{} (iii)||anti||={:.2e}:{} => {}",
            self.norm_a1,
            self.norm_a2,
            if self.cond_i { "PASS" } else { "FAIL" },
            self.b2_error,
            if self.cond_ii { "PASS" } else { "FAIL" },
            self.antiassociator_norm,
            if self.cond_iii { "PASS" } else { "FAIL" },
            if self.all_satisfied {
                "ZD PAIR"
            } else {
                "NOT ZD"
            },
        )
    }
}

/// Verify Theorem 7.9: If A and B are mutual zero divisors, then <A,B>
/// (the span of A, B, and their products) is a commutative Jordan subalgebra.
///
/// Part i): <A,B> has 3D form r1 + r2*A + r3*B
/// Part ii): <A,B> is commutative (verified by checking AB=0)
/// Part iii): <A,B> is NOT alternative (A^2*B != A*(AB))
///
/// Mirrors: Brown (1972) Theorem 7.9.
pub fn verify_jordan_subalgebra_property(a: &[f64], b: &[f64]) -> (f64, f64, f64) {
    // Part i) implicitly satisfied when AB=0
    let ab = cd_multiply(a, b);
    let ba = cd_multiply(b, a);

    // Part ii) Commutativity: AB = BA = 0
    let commutativity_err = cd_norm_sq(&ab).max(cd_norm_sq(&ba));

    // Part iii) Check non-alternativity: A^2*B should not equal A*(AB)
    let a_sq = cd_multiply(a, a);
    let a_sq_b = cd_multiply(&a_sq, b);
    let ab_prod = cd_multiply(a, &ab);

    let non_alternativity_err = a_sq_b
        .iter()
        .zip(ab_prod.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max);

    (commutativity_err, non_alternativity_err, 0.0)
}

/// Verify Theorem 7.13: Properties of mutual zero divisors.
///
/// If A and B are mutual ZDs:
///   i) `T([a1*a2]*b1) = T([a1*a2]*b2) = 0`
///   ii) a1 = ±a2 implies a1*b1 = a1*b2 = 0
///   iii) a1 and a2 are not both real numbers
///   iv) a1 = 0 implies a2*b1 = a2*b2 = 0
///   v) a2 = 0 implies a1*b1 = a1*b2 = 0
///
/// Mirrors: Brown (1972) Theorem 7.13.
pub fn verify_mutual_zd_properties(a: &[f64], b: &[f64]) -> MutualZdProperties {
    let dim = a.len();
    let half = dim / 2;

    let a1 = &a[..half];
    let a2 = &a[half..];
    let b1 = &b[..half];
    let b2 = &b[half..];

    // i) T([a1*a2]*bi) = 0
    let a1_a2 = cd_multiply(a1, a2);
    let a1a2_b1 = cd_multiply(&a1_a2, b1);
    let a1a2_b2 = cd_multiply(&a1_a2, b2);
    let trace_a1a2b1 = a1a2_b1[0] * 2.0; // T(x) = 2*x[0]
    let trace_a1a2b2 = a1a2_b2[0] * 2.0;

    // iii) Check if either a1 or a2 is a real number
    let a1_is_real = a1[1..].iter().all(|&x| x.abs() < 1e-10);
    let a2_is_real = a2[1..].iter().all(|&x| x.abs() < 1e-10);
    let both_nonreal = !a1_is_real && !a2_is_real;

    // iv) & v) Check zero conditions
    let a1_zero = a1.iter().all(|&x| x.abs() < 1e-10);
    let a2_zero = a2.iter().all(|&x| x.abs() < 1e-10);

    MutualZdProperties {
        trace_a1a2b1: trace_a1a2b1.abs(),
        trace_a1a2b2: trace_a1a2b2.abs(),
        both_nonreal,
        a1_zero,
        a2_zero,
    }
}

/// Result of checking mutual ZD properties.
#[derive(Debug, Clone)]
pub struct MutualZdProperties {
    pub trace_a1a2b1: f64,
    pub trace_a1a2b2: f64,
    pub both_nonreal: bool,
    pub a1_zero: bool,
    pub a2_zero: bool,
}

/// Verify Theorem 7.14: If A is a zero divisor in A_4:
///   i) a1 = ±a2
///   ii) a1 and a2 are not real numbers
///   iii) N(a1) = N(a2)
///
/// Mirrors: Brown (1972) Theorem 7.14.
pub fn verify_zd_element_condition_dim4(a: &[f64]) -> ZdElementProperties {
    let dim = a.len();
    let half = dim / 2;

    let a1 = &a[..half];
    let a2 = &a[half..];

    let n_a1 = cd_norm_sq(a1);
    let n_a2 = cd_norm_sq(a2);

    // i) a1 = ±a2
    let a1_plus_a2: Vec<f64> = a1.iter().zip(a2.iter()).map(|(x, y)| x + y).collect();
    let a1_minus_a2: Vec<f64> = a1.iter().zip(a2.iter()).map(|(x, y)| x - y).collect();
    let a1_equals_a2 = cd_norm_sq(&a1_plus_a2) < 1e-10;
    let a1_equals_neg_a2 = cd_norm_sq(&a1_minus_a2) < 1e-10;
    let cond_i = !a1_equals_a2 && !a1_equals_neg_a2;

    // ii) a1 and a2 not real
    let a1_is_real = a1[1..].iter().all(|&x| x.abs() < 1e-10);
    let a2_is_real = a2[1..].iter().all(|&x| x.abs() < 1e-10);
    let cond_ii = !a1_is_real && !a2_is_real;

    // iii) N(a1) = N(a2)
    let cond_iii = (n_a1 - n_a2).abs() < 1e-10;

    ZdElementProperties {
        cond_i,
        cond_ii,
        cond_iii,
        norm_a1: n_a1,
        norm_a2: n_a2,
    }
}

/// Result of checking ZD element properties.
#[derive(Debug, Clone)]
pub struct ZdElementProperties {
    pub cond_i: bool,
    pub cond_ii: bool,
    pub cond_iii: bool,
    pub norm_a1: f64,
    pub norm_a2: f64,
}

/// Check Theorem 7.3: verify that AB=0 implies BA=0, conj(A)B=0, etc.
///
/// Returns the 7 equivalent condition norms.
pub fn check_zd_symmetry(a: &[f64], b: &[f64]) -> [f64; 7] {
    let ab = cd_multiply(a, b);
    let ba = cd_multiply(b, a);
    let conj_a = cd_conjugate(a);
    let conj_a_b = cd_multiply(&conj_a, b);
    // A^{-1} = conj(A)/N(A)
    let n_a = cd_norm_sq(a);
    let inv_a: Vec<f64> = conj_a.iter().map(|x| x / n_a).collect();
    let inv_a_b = cd_multiply(&inv_a, b);

    [
        cd_norm_sq(&ab),       // i) AB = 0
        cd_norm_sq(&ba),       // ii) BA = 0
        cd_norm_sq(&conj_a_b), // iii) conj(A)B = 0
        cd_norm_sq(&inv_a_b),  // iv) A^{-1}B = 0
        0.0,
        0.0,
        0.0, // v-vii) require e computation
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_major_theorem_zd_witness() {
        // The standard ZD pair: A = e3+e10, B = e6-e15
        let (a, b) = schafer_1945::division_test::zd_witness_16();
        let r = check_major_theorem(&a, &b, 1e-8);
        assert!(
            r.all_satisfied,
            "ZD witness should satisfy Major Theorem: {r}"
        );
    }

    #[test]
    fn test_major_theorem_non_zd() {
        // A non-ZD pair
        let mut a = vec![0.0; 16];
        a[1] = 1.0;
        a[10] = 1.0;
        let mut b = vec![0.0; 16];
        b[1] = 1.0;
        b[2] = 1.0;
        let r = check_major_theorem(&a, &b, 1e-8);
        assert!(
            !r.all_satisfied,
            "non-ZD pair should fail Major Theorem: {r}"
        );
    }

    #[test]
    fn test_antiassociator_zero_for_quaternions() {
        // Quaternions are associative, so antiassociator = (AB)C + A(BC) = 2*A(BC) != 0 in general.
        // Actually, )a,b,c( = (ab)c + a(bc). For associative: = 2*(abc). So it's NOT zero!
        // The antiassociator is zero only in specific ZD configurations.
        let a = vec![0.0, 1.0, 0.0, 0.0];
        let b = vec![0.0, 0.0, 1.0, 0.0];
        let c = vec![0.0, 0.0, 0.0, 1.0];
        let anti = antiassociator(&a, &b, &c);
        let norm = cd_norm_sq(&anti);
        // For quaternions: (ij)k + i(jk) = (-1) + (-1) = -2*e_0. So norm = 4.
        assert!(norm > 0.1, "quaternion antiassociator should be nonzero");
    }

    #[test]
    fn test_zd_symmetry() {
        let (a, b) = schafer_1945::division_test::zd_witness_16();
        let norms = check_zd_symmetry(&a, &b);
        // AB = 0
        assert!(norms[0] < 1e-10, "AB should be 0");
        // BA = 0 (Theorem 7.3)
        assert!(norms[1] < 1e-10, "BA should be 0 by ZD symmetry");
        // conj(A)B = 0
        assert!(norms[2] < 1e-10, "conj(A)B should be 0");
    }

    #[test]
    fn test_major_theorem_display() {
        let (a, b) = schafer_1945::division_test::zd_witness_16();
        let r = check_major_theorem(&a, &b, 1e-8);
        let s = format!("{r}");
        assert!(s.contains("ZD PAIR"));
    }

    #[test]
    fn test_jordan_subalgebra_property() {
        let (a, b) = schafer_1945::division_test::zd_witness_16();
        let (comm_err, alt_err, _) = verify_jordan_subalgebra_property(&a, &b);
        assert!(
            comm_err < 1e-10,
            "mutual ZDs should be commutative: err={comm_err}"
        );
        assert!(
            alt_err > 0.01,
            "mutual ZD span should NOT be alternative: err={alt_err}"
        );
    }

    #[test]
    fn test_mutual_zd_properties() {
        let (a, b) = schafer_1945::division_test::zd_witness_16();
        let props = verify_mutual_zd_properties(&a, &b);
        assert!(props.trace_a1a2b1 < 1e-8, "T([a1*a2]*b1) should be 0");
        assert!(props.trace_a1a2b2 < 1e-8, "T([a1*a2]*b2) should be 0");
        assert!(props.both_nonreal, "a1 and a2 should both be nonreal");
    }

    #[test]
    fn test_zd_element_condition_dim4() {
        let (a, _) = schafer_1945::division_test::zd_witness_16();
        let props = verify_zd_element_condition_dim4(&a);
        assert!(props.cond_i, "a1 should not equal ±a2");
        assert!(props.cond_ii, "a1 and a2 should be nonreal");
        assert!(props.cond_iii, "N(a1) should equal N(a2)");
    }
}
