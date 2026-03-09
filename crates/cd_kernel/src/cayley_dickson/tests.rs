use super::*;

#[test]
fn test_real_multiply() {
    let result = cd_multiply(&[3.0], &[5.0]);
    assert_eq!(result, vec![15.0]);
}

#[test]
fn test_complex_multiply() {
    let result = cd_multiply(&[1.0, 2.0], &[3.0, 4.0]);
    assert!((result[0] + 5.0).abs() < 1e-10);
    assert!((result[1] - 10.0).abs() < 1e-10);
}

#[test]
fn test_find_zero_divisors_octonion_none() {
    let results = find_zero_divisors(8, 1e-10);
    assert!(results.is_empty());
}

#[test]
fn test_find_zero_divisors_sedenion() {
    let results = find_zero_divisors(16, 1e-10);
    assert!(!results.is_empty());
}

#[test]
fn test_simd_matches_scalar() {
    for dim in [4, 8, 16, 32] {
        let a: Vec<f64> = (0..dim).map(|i| (i as f64 * 0.123).sin()).collect();
        let b: Vec<f64> = (0..dim).map(|i| (i as f64 * 0.456).cos()).collect();
        let scalar_result = cd_multiply(&a, &b);
        let simd_result = cd_multiply_simd(&a, &b);
        for (s, m) in scalar_result.iter().zip(&simd_result) {
            assert!((s - m).abs() < 1e-12);
        }
    }
}

#[test]
fn test_sign_table_matches_iter() {
    let table = SignTable::new(16);
    for p in 0..16 {
        for q in 0..16 {
            assert_eq!(table.sign(p, q), cd_basis_mul_sign_iter(16, p, q));
        }
    }
}

#[test]
fn test_split_signature_matches_standard() {
    let sig = CdSignature::standard(8);
    for p in 0..8 {
        for q in 0..8 {
            assert_eq!(
                cd_basis_mul_sign(8, p, q),
                cd_basis_mul_sign_split(8, p, q, &sig)
            );
        }
    }
}

#[test]
fn test_associator_density_quaternions() {
    let (density, failures) = measure_associator_density(4, 200, 42, 1e-8);
    assert_eq!(failures, 0);
    assert!((density - 0.0).abs() < 1e-10);
}

#[test]
fn test_associator_density_sedenions() {
    let (density, failures) = measure_associator_density(16, 200, 42, 1e-8);
    assert!(density > 90.0);
    assert!(failures > 150);
}

#[test]
fn test_associator_stats_large_dim() {
    let stats = associator_independence_stats(16, 200, 42);
    assert!(stats.mean_assoc_sq.is_finite());
}
