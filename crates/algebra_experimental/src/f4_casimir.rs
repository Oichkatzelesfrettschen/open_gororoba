/// Computes the exact F4 Casimir Invariant Ratio
///
/// From the Unified Monograph:
/// "Theorem 12.6 (F4 Casimir Eigenvalue -- Rocq C-035): epsilon = C2(26)/|Delta+(F4)| = 6/24 = 1/4 exactly."
///
/// This module implements the exact Lie algebra root system combinatorics for F4
/// in pure Rust, generating the positive roots, the Weyl vector, and computing the
/// exact quadratic Casimir invariant of the 26-dimensional fundamental representation.
pub fn compute_f4_casimir_ratio() -> (f64, usize, f64) {
    // 1. Generate the 24 positive roots of F4 in R^4
    // F4 roots:
    // - e_i +/- e_j (1 <= i < j <= 4) -> 12 positive roots (where i < j, so e_i has positive coeff)
    // - e_i (1 <= i <= 4) -> 4 positive roots
    // - 1/2 (e_1 +/- e_2 +/- e_3 +/- e_4) -> 8 positive roots (where e_1 is positive)

    let mut positive_roots: Vec<[f64; 4]> = Vec::new();

    // e_i +/- e_j
    for i in 0..4 {
        for j in (i + 1)..4 {
            let mut r1 = [0.0; 4];
            r1[i] = 1.0;
            r1[j] = 1.0;
            positive_roots.push(r1);

            let mut r2 = [0.0; 4];
            r2[i] = 1.0;
            r2[j] = -1.0;
            positive_roots.push(r2);
        }
    }

    // e_i
    for i in 0..4 {
        let mut r = [0.0; 4];
        r[i] = 1.0;
        positive_roots.push(r);
    }

    // 1/2 (e_1 +/- e_2 +/- e_3 +/- e_4)
    // e_1 is positive.
    for s2 in [-1.0, 1.0] {
        for s3 in [-1.0, 1.0] {
            for s4 in [-1.0, 1.0] {
                positive_roots.push([0.5, 0.5 * s2, 0.5 * s3, 0.5 * s4]);
            }
        }
    }

    let num_positive_roots = positive_roots.len(); // Must be 24

    // 2. Compute the Weyl vector (rho) = 1/2 sum(positive_roots)
    let mut rho = [0.0; 4];
    for root in &positive_roots {
        for i in 0..4 {
            rho[i] += 0.5 * root[i];
        }
    }

    // For F4, rho is exactly [11/2, 5/2, 3/2, 1/2]

    // 3. Compute the Quadratic Casimir C2(lambda) for the 26D representation.
    // The highest weight lambda of the 26-dimensional fundamental rep of F4 is exactly e_1 = [1, 0, 0, 0]
    let lambda = [1.0, 0.0, 0.0, 0.0];

    // C2(lambda) = <lambda + 2*rho, lambda>
    let mut c2_standard = 0.0;
    for i in 0..4 {
        let val = lambda[i] + 2.0 * rho[i];
        c2_standard += val * lambda[i];
    }

    // Physics Normalization:
    // Standard Lie Algebra convention assigns long roots a squared length of 2.
    // The Monograph / Physics normalization assigns long roots a squared length of 1.
    // Therefore, the Casimir eigenvalue is scaled by 1/2.
    let c2 = c2_standard * 0.5;

    // The ratio epsilon
    let epsilon = c2 / (num_positive_roots as f64);

    (c2, num_positive_roots, epsilon)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f4_casimir_ratio() {
        println!("--- EXACT F4 LIE ALGEBRA CASIMIR COMPUTATION ---");
        let (c2, num_pos_roots, epsilon) = compute_f4_casimir_ratio();

        println!("Quadratic Casimir C2(26): {}", c2);
        println!("Positive Roots |Delta+(F4)|: {}", num_pos_roots);
        println!("Ratio Epsilon: {:.4}", epsilon);

        assert_eq!(num_pos_roots, 24, "F4 must have exactly 24 positive roots");
        assert!((c2 - 6.0).abs() < 1e-9, "C2(26) must be exactly 6");
        assert!((epsilon - 0.25).abs() < 1e-9, "Epsilon must be exactly 1/4");

        println!("<EMOJI+2705> Theorem 12.6 Verified: epsilon = 1/4 EXACTLY.");
    }
}
