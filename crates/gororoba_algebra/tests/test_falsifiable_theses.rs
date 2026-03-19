//! Part IV: Falsifiable Theses Verification
//!
//! This module programmatically implements the falsification tests proposed in
//! the "COMPREHENSIVE AUDIT, SYNTHESIS, AND ROADMAP" document.
//!
//! # Number Theory Theses
//!
//! ## Thesis 1: No normed division algebra exists in dimension 16.
//! - **Test**: Construct norm on sedenions with `||ab|| = ||a|| * ||b||`
//! - **Prediction**: Impossible (zero divisors prevent this)
//! - **Falsification Criteria**: Find such a norm -> Contradicts Hurwitz (1898)
//!
//! ## Thesis 2: Associativity lost permanently beyond quaternions.
//! - **Test**: Check `(ab)c = a(bc)` for octonions, sedenions.
//! - **Prediction**: Always fails for some elements.
//! - **Falsification Criteria**: Find an associative 8D+ Cayley-Dickson algebra.

use cd_kernel::cayley_dickson::{
    cd_multiply, cd_norm_sq,
    sedenion::sedenion_multiply_explicit,
};

/// Mathematical Thesis 1: Sedenions violate the division algebra property.
/// We verify this by concretely exhibiting zero divisors a and b where
/// ||a|| != 0 and ||b|| != 0, but ||a * b|| == 0.
#[test]
fn test_thesis_1_sedenion_no_normed_division() {
    // We construct the canonical sedenion zero divisor pair:
    // a = e_1 + e_10
    // b = e_15 - e_4
    let mut a = vec![0.0; 16];
    let mut b = vec![0.0; 16];

    a[1] = 1.0;
    a[10] = 1.0;

    b[15] = 1.0;
    b[4] = -1.0;

    let norm_sq_a = cd_norm_sq(&a);
    let norm_sq_b = cd_norm_sq(&b);
    
    assert!(norm_sq_a > 1e-10, "a must be non-zero");
    assert!(norm_sq_b > 1e-10, "b must be non-zero");

    let product = cd_multiply(&a, &b);
    let norm_sq_prod = cd_norm_sq(&product);

    // If it were a normed division algebra, ||ab||^2 = ||a||^2 * ||b||^2
    // For Sedenions with these elements, the product is zero!
    assert!(norm_sq_prod < 1e-10, "Product must be zero, exhibiting zero divisors");
    
    // Thus ||ab|| != ||a|| * ||b||
    assert!(
        (norm_sq_prod - (norm_sq_a * norm_sq_b)).abs() > 1e-10,
        "Norm composition property MUST FAIL for zero divisors!"
    );
}

/// Mathematical Thesis 2: Associativity is lost permanently beyond quaternions.
/// We verify this by explicitly finding (ab)c != a(bc) in the Octonions (8D)
/// and Sedenions (16D).
#[test]
fn test_thesis_2_associativity_loss() {
    // For octonions, we check e_1, e_2, e_4 which form an anti-associative triad.
    let mut e1 = vec![0.0; 8];
    e1[1] = 1.0;
    let mut e2 = vec![0.0; 8];
    e2[2] = 1.0;
    let mut e4 = vec![0.0; 8];
    e4[4] = 1.0;

    let e1_e2 = cd_multiply(&e1, &e2);
    let left_assoc = cd_multiply(&e1_e2, &e4);

    let e2_e4 = cd_multiply(&e2, &e4);
    let right_assoc = cd_multiply(&e1, &e2_e4);

    // They should be anti-associative: (e_1 e_2) e_4 = - e_1 (e_2 e_4)
    let mut difference_found = false;
    for i in 0..8 {
        if (left_assoc[i] - right_assoc[i]).abs() > 1e-10 {
            difference_found = true;
            break;
        }
    }
    assert!(difference_found, "Associativity MUST FAIL in octonions (8D)");

    // The same holds for sedenions.
    let mut s1 = [0.0; 16];
    s1[1] = 1.0;
    let mut s2 = [0.0; 16];
    s2[2] = 1.0;
    let mut s4 = [0.0; 16];
    s4[4] = 1.0;

    let s1_s2 = sedenion_multiply_explicit(&s1, &s2);
    let s_left = sedenion_multiply_explicit(&s1_s2, &s4);
    let s2_s4 = sedenion_multiply_explicit(&s2, &s4);
    let s_right = sedenion_multiply_explicit(&s1, &s2_s4);

    let mut difference_sedenion = false;
    for i in 0..16 {
        if (s_left[i] - s_right[i]).abs() > 1e-10 {
            difference_sedenion = true;
            break;
        }
    }
    assert!(difference_sedenion, "Associativity MUST FAIL in sedenions (16D)");
}

/// Verify Clifford structure aliases align with Synthesis.
#[test]
fn test_clifford_taxonomy() {
    use gororoba_algebra::construction::clifford::CliffordSignature;
    
    // Cl(0,1) ≅ C
    let c = CliffordSignature::complex();
    assert_eq!(c.dim(), 2);
    assert_eq!(c.basis_square(1), -1);

    // Cl(0,2) ≅ H
    let h = CliffordSignature::quaternions();
    assert_eq!(h.dim(), 4);
    assert_eq!(h.basis_square(1), -1);
    assert_eq!(h.basis_square(2), -1);

    // Cl(1,3) ≅ Spacetime
    let st = CliffordSignature::spacetime();
    assert_eq!(st.dim(), 16);
    assert_eq!(st.basis_square(1), 1); // e1^2 = +1
    assert_eq!(st.basis_square(2), -1); // e2^2 = -1
    assert_eq!(st.basis_square(3), -1);
    assert_eq!(st.basis_square(4), -1);
}
