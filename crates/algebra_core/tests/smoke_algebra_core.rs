use algebra_core::{cd_conjugate, cd_multiply, cd_norm_sq, find_zero_divisors};

#[test]
fn smoke_cd_multiply_and_conjugate_are_well_formed() {
    let a = vec![1.0, 2.0, 0.0, -1.0];
    let b = vec![0.5, -1.0, 3.0, 0.25];

    let product = cd_multiply(&a, &b);
    let conjugate = cd_conjugate(&a);

    assert_eq!(product.len(), a.len());
    assert_eq!(conjugate, vec![1.0, -2.0, -0.0, 1.0]);
    assert!(cd_norm_sq(&product).is_finite());
}

#[test]
fn smoke_sedenion_zero_divisor_search_finds_pairs() {
    let pairs = find_zero_divisors(16, 1e-10);
    assert!(
        !pairs.is_empty(),
        "expected canonical sedenion zero divisors"
    );
}
