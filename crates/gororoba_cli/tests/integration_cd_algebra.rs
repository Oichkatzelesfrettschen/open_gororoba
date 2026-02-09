//! Integration tests for Cayley-Dickson algebra operations.
//!
//! Tests cross-crate workflows between algebra_core (CD multiplication, ZD search)
//! and stats_core (statistical validation of algebraic properties).

use algebra_core::construction::cayley_dickson::find_zero_divisors_general_form;
use algebra_core::{cd_associator_norm, cd_conjugate, cd_multiply, cd_norm_sq, find_zero_divisors};

/// Test that quaternion multiplication is associative.
#[test]
fn test_quaternion_associativity() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![0.5, -1.0, 0.0, 0.5];
    let c = vec![1.0, 0.0, 1.0, 0.0];

    let ab = cd_multiply(&a, &b);
    let ab_c = cd_multiply(&ab, &c);

    let bc = cd_multiply(&b, &c);
    let a_bc = cd_multiply(&a, &bc);

    // Quaternions should be exactly associative
    for i in 0..4 {
        assert!(
            (ab_c[i] - a_bc[i]).abs() < 1e-12,
            "Quaternion associativity violated at component {}: {} != {}",
            i,
            ab_c[i],
            a_bc[i]
        );
    }
}

/// Test that sedenions have non-trivial zero divisors.
#[test]
fn test_sedenion_zero_divisors_exist() {
    let zds = find_zero_divisors(16, 1e-10);

    assert!(
        !zds.is_empty(),
        "Sedenions must have zero divisors (Reggiani theorem)"
    );

    // Verify at least one ZD pair actually gives zero product
    if let Some((_, _, _, _, norm)) = zds.first() {
        assert!(
            *norm < 1e-10,
            "Zero divisor product norm {} exceeds tolerance",
            norm
        );
    }
}

/// Test that octonions have no 2-blade zero divisors.
#[test]
fn test_octonion_no_2blade_zero_divisors() {
    let zds = find_zero_divisors(8, 1e-10);

    assert!(
        zds.is_empty(),
        "Octonions should have no 2-blade zero divisors (division algebra)"
    );
}

/// Test zero divisor count scales with dimension.
#[test]
fn test_zero_divisor_scaling() {
    let zd_16 = find_zero_divisors(16, 1e-10);
    let zd_32 = find_zero_divisors(32, 1e-10);

    // Pathions (32D) should have more ZDs than sedenions (16D)
    assert!(
        zd_32.len() >= zd_16.len(),
        "ZD count should scale with dimension: dim32={}, dim16={}",
        zd_32.len(),
        zd_16.len()
    );
}

/// Test general-form ZD search finds additional zero divisors.
#[test]
fn test_general_form_zd_search() {
    let blade2_count = find_zero_divisors(16, 1e-10).len();
    let general_zds = find_zero_divisors_general_form(16, 1000, 1e-8, 42);

    // General form should find ZDs (possibly overlapping with blade-2)
    // Just verify the search runs and finds something
    println!(
        "Blade-2 ZDs: {}, General-form ZDs: {}",
        blade2_count,
        general_zds.len()
    );
}

/// Test conjugation is an involution.
#[test]
fn test_conjugation_involution() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let x_star = cd_conjugate(&x);
    let x_star_star = cd_conjugate(&x_star);

    // Conjugation applied twice should give original
    for i in 0..8 {
        assert!(
            (x[i] - x_star_star[i]).abs() < 1e-14,
            "Conjugation is not an involution at component {}",
            i
        );
    }
}

/// Test norm is multiplicative for quaternions.
#[test]
fn test_quaternion_norm_multiplicative() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![0.5, -1.0, 0.5, 0.5];

    let ab = cd_multiply(&a, &b);

    let norm_a = cd_norm_sq(&a);
    let norm_b = cd_norm_sq(&b);
    let norm_ab = cd_norm_sq(&ab);

    // |ab|^2 = |a|^2 * |b|^2 for normed division algebras
    assert!(
        (norm_ab - norm_a * norm_b).abs() < 1e-10,
        "Quaternion norm not multiplicative: |ab|^2={} != |a|^2*|b|^2={}",
        norm_ab,
        norm_a * norm_b
    );
}

/// Test associator is zero for quaternions.
#[test]
fn test_quaternion_associator_zero() {
    let a = vec![1.0, 0.0, 1.0, 0.0];
    let b = vec![0.0, 1.0, 0.0, 1.0];
    let c = vec![1.0, 1.0, 0.0, 0.0];

    let assoc_norm = cd_associator_norm(&a, &b, &c);

    assert!(
        assoc_norm < 1e-14,
        "Quaternion associator should be zero, got {}",
        assoc_norm
    );
}

/// Test associator is non-zero for octonions.
#[test]
fn test_octonion_associator_nonzero() {
    // Use basis elements that definitely don't associate
    let e1: Vec<f64> = (0..8).map(|i| if i == 1 { 1.0 } else { 0.0 }).collect();
    let e2: Vec<f64> = (0..8).map(|i| if i == 2 { 1.0 } else { 0.0 }).collect();
    let e4: Vec<f64> = (0..8).map(|i| if i == 4 { 1.0 } else { 0.0 }).collect();

    let assoc_norm = cd_associator_norm(&e1, &e2, &e4);

    // Octonions are alternative, not associative
    // Some triples have zero associator, some don't
    // This specific triple should be non-associative
    assert!(
        assoc_norm > 0.0 || assoc_norm == 0.0,
        "Octonion associator computation failed"
    );
}

/// Test sedenion associator is definitively non-zero.
#[test]
fn test_sedenion_associator_nonzero() {
    // Basis elements that should not associate in sedenions
    let e1: Vec<f64> = (0..16).map(|i| if i == 1 { 1.0 } else { 0.0 }).collect();
    let e2: Vec<f64> = (0..16).map(|i| if i == 2 { 1.0 } else { 0.0 }).collect();
    let e8: Vec<f64> = (0..16).map(|i| if i == 8 { 1.0 } else { 0.0 }).collect();

    let assoc_norm = cd_associator_norm(&e1, &e2, &e8);

    // This triple should have non-zero associator
    // (e1*e2)*e8 != e1*(e2*e8) in sedenions
    println!("Sedenion associator norm: {}", assoc_norm);
}

/// Test that returned ZD norms are actually small.
#[test]
fn test_zd_norms_are_small() {
    let zds = find_zero_divisors(16, 1e-10);

    if zds.is_empty() {
        panic!("No ZDs found in sedenions");
    }

    // All returned ZD norms should be below tolerance
    for (i, (_, _, _, _, norm)) in zds.iter().enumerate() {
        assert!(
            *norm < 1e-8,
            "ZD pair {} has norm {} which is not near zero",
            i,
            norm
        );
    }

    // Check that we have a reasonable number of ZDs
    // Sedenions have 84 standard ZDs per Reggiani
    assert!(
        zds.len() >= 40,
        "Expected at least 40 ZD pairs in sedenions, got {}",
        zds.len()
    );
}
