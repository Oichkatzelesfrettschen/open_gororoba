//! Cross-validation: verified_core refuted-claims modules vs production crates.
//!
//! Confirms that the Rocq-verified arithmetic produces identical results
//! to the corresponding production implementations in algebra_core and
//! cosmology_core.

const TOL: f64 = 1e-12;

// --- C-883: Spectral dimension cross-validation ---

#[test]
fn cross_validate_calcagni_spectral_dimension() {
    // verified_core::spectral_dim::calcagni_d_s(s) should match
    // cosmology_core::spectral::calcagni_spectral_dimension(k, alpha)
    // when s = (k/k_star)^{-alpha} and k_star = 1.
    //
    // At k=1.0, alpha=1.0: s = 1.0^{-1.0} = 1.0, d_s = 3.0
    let vc_result = verified_core::spectral_dim::calcagni_d_s(1.0);
    let cc_result = cosmology_core::spectral::calcagni_spectral_dimension(1.0, 1.0);
    assert!(
        (vc_result - cc_result).abs() < TOL,
        "spectral dim mismatch at s=1: vc={vc_result}, cc={cc_result}"
    );

    // At k=2.0, alpha=1.0: s = 2.0^{-1.0} = 0.5
    let s = 0.5_f64;
    let vc_result = verified_core::spectral_dim::calcagni_d_s(s);
    let cc_result = cosmology_core::spectral::calcagni_spectral_dimension(2.0, 1.0);
    assert!(
        (vc_result - cc_result).abs() < TOL,
        "spectral dim mismatch at s=0.5: vc={vc_result}, cc={cc_result}"
    );

    // At k=0.5, alpha=2.0: s = 0.5^{-2.0} = 4.0
    let s = 4.0_f64;
    let vc_result = verified_core::spectral_dim::calcagni_d_s(s);
    let cc_result = cosmology_core::spectral::calcagni_spectral_dimension(0.5, 2.0);
    assert!(
        (vc_result - cc_result).abs() < TOL,
        "spectral dim mismatch at s=4: vc={vc_result}, cc={cc_result}"
    );
}

// --- C-881: GF(2) separation cross-validation ---

#[test]
fn cross_validate_gf2_separation() {
    use algebra_core::analysis::{
        boxkites::motif_components_for_cross_assessors,
        projective_geometry::{component_xor_label, find_boolean_class_predicate},
    };

    // Get production PG(3,2) data
    let comps = motif_components_for_cross_assessors(32);
    let labels: Vec<usize> = comps
        .iter()
        .map(|c| component_xor_label(c).unwrap())
        .collect();
    let classes: Vec<usize> = comps
        .iter()
        .map(|c| if c.edges.len() == 84 { 0 } else { 1 })
        .collect();

    // Verify label/class data matches hardcoded constants
    let vc_labels = verified_core::gf2::PG32_LABELS;
    let vc_classes = verified_core::gf2::PG32_CLASSES;
    for (i, (&l, &vc_l)) in labels.iter().zip(vc_labels.iter()).enumerate() {
        assert_eq!(
            l, vc_l as usize,
            "label mismatch at index {i}: production={l}, verified={vc_l}"
        );
    }
    // In verified_core, true = heptacross (class 0), false = mixed (class 1)
    // This is flipped relative to the production code's integer classes
    for (i, (&c, &vc_c)) in classes.iter().zip(vc_classes.iter()).enumerate() {
        let expected = c == 0; // heptacross (class 0) maps to true
        assert_eq!(
            expected, vc_c,
            "class mismatch at index {i}: production={c}, verified={vc_c}"
        );
    }

    // Verify cubic witness separates in both implementations
    let n_bits = 4;
    let prod_result = find_boolean_class_predicate(&labels, &classes, n_bits, 3);
    assert!(
        prod_result.is_some(),
        "production should find cubic separator"
    );

    assert!(
        verified_core::gf2::separates(&vc_labels, &vc_classes, &verified_core::gf2::CUBIC_WITNESS),
        "verified cubic witness should separate"
    );

    // Verify degree 1 and 2 fail in both
    assert!(
        find_boolean_class_predicate(&labels, &classes, n_bits, 1).is_none(),
        "degree 1 should fail in production"
    );
    assert!(
        find_boolean_class_predicate(&labels, &classes, n_bits, 2).is_none(),
        "degree 2 should fail in production"
    );
}

// --- C-884: Neg-dim degeneracy self-consistency ---

#[test]
fn cross_validate_neg_dim_degeneracy() {
    use verified_core::neg_dim::{neg_dim_product, neg_dim_w_eff};

    // Same product => same w_eff (the formal theorem)
    let pairs: &[(f64, f64)] = &[(0.0, 1.0), (1.0, 0.6), (2.0, 3.0 / 7.0), (-0.5, 1.5)];

    let target_product = neg_dim_product(0.0, 1.0); // = 1.5
    let target_w = neg_dim_w_eff(0.0, 1.0);

    for &(a, e) in pairs {
        let p = neg_dim_product(a, e);
        if (p - target_product).abs() < TOL {
            let w = neg_dim_w_eff(a, e);
            assert!(
                (w - target_w).abs() < TOL,
                "same product should give same w_eff: ({a},{e}) -> w={w}, expected {target_w}"
            );
        }
    }
}

// --- C-882: Parity-clique edge counts ---

#[test]
fn cross_validate_parity_clique() {
    use verified_core::parity_clique::clique_edges;

    // Known formula: n*(n-1)/2
    for n in 0_usize..20 {
        assert_eq!(clique_edges(n), n * n.saturating_sub(1) / 2);
    }

    // Specific dim-16 and dim-32 values
    assert_eq!(2 * clique_edges(4), 12);
    assert_eq!(2 * clique_edges(8), 56);
}

// --- C-885: Democratic mixing self-consistency ---

#[test]
fn cross_validate_democratic_mixing() {
    use verified_core::s3_mixing::{is_democratic, is_s3_invariant};

    // J_3 / 3 is both S3-invariant and democratic
    let j3 = [[1.0 / 3.0; 3]; 3];
    assert!(is_s3_invariant(&j3, 1e-15));
    assert!(is_democratic(&j3, 1e-15));

    // Identity matrix is S3-invariant but not democratic
    let eye = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    assert!(is_s3_invariant(&eye, 1e-15));
    assert!(!is_democratic(&eye, 1e-10));

    // Random non-symmetric matrix is neither
    let bad = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
    assert!(!is_s3_invariant(&bad, 1e-10));
}
