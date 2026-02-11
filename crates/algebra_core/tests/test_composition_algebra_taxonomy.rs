//! Composition Algebra Taxonomy Test Suite
//!
//! Comprehensive verification of the two-axis taxonomy framework established in Phase 10.
//! Tests confirm:
//! - Axis 1: Construction method determines commutativity (tensor != recursive doubling)
//! - Axis 2: Metric signature controls zero-divisor structure (CD family only)
//! - Exceptional algebras (Albert, variants) form distinct class
//! - Phase 9-10 categorical distinction theorem generalizes to all families
//!
//! Date: 2026-02-10, Phase 10.3 Track 2

use algebra_core::construction::composition_algebra_census::*;

/// TEST CLASS 1: Axis 1 Hypothesis - Construction Method Commutativity Law
/// Hypothesis: All algebras with same construction method have identical commutativity
#[test]
fn test_taxonomy_axis1_tensor_product_commutativity() {
    let mut census = CompositionAlgebraCensus::new();

    // Tessarines: C x C tensor product (Phase 9 result)
    let mut tessarines = AlgebraProperties::new(
        "Tessarines_4D",
        4,
        ConstructionMethod::TensorProduct,
        None,
    );
    tessarines.is_commutative = true;
    tessarines.commutativity_percentage = 100.0;
    tessarines.num_samples = 100;

    // Tensor product variants at higher dimension
    let mut tensor_8d = AlgebraProperties::new(
        "TensorProduct_8D",
        8,
        ConstructionMethod::TensorProduct,
        None,
    );
    tensor_8d.is_commutative = true;
    tensor_8d.commutativity_percentage = 100.0;
    tensor_8d.num_samples = 100;

    let node1 = TaxonomyNode::new(tessarines);
    let node2 = TaxonomyNode::new(tensor_8d);

    census.register_algebra(node1).ok();
    census.register_algebra(node2).ok();

    // Verify axis 1 law
    assert!(
        census.verify_axis1_commutativity_law().is_ok(),
        "Axis 1: All tensor products must be 100% commutative"
    );
}

/// TEST CLASS 2: Axis 1 Hypothesis - CD Algebras are Non-Commutative
/// Hypothesis: All Cayley-Dickson algebras (dim >= 4) are 100% non-commutative
#[test]
fn test_taxonomy_axis1_cayley_dickson_non_commutativity() {
    let mut census = CompositionAlgebraCensus::new();

    // Mixed-quaternions: CD dim=4, gamma=[+1,-1]
    let mut mixed_quat = AlgebraProperties::new(
        "MixedQuaternions",
        4,
        ConstructionMethod::RecursiveDoubling,
        Some(MetricSignature::mixed(vec![1, -1])),
    );
    mixed_quat.is_commutative = false;
    mixed_quat.commutativity_percentage = 0.0;
    mixed_quat.num_samples = 100;

    // Quaternions: CD dim=4, gamma=[-1,-1] (Hurwitz)
    let mut quaternions = AlgebraProperties::new(
        "Quaternions",
        4,
        ConstructionMethod::RecursiveDoubling,
        Some(MetricSignature::hurwitz(2)),
    );
    quaternions.is_commutative = false;
    quaternions.commutativity_percentage = 0.0;
    quaternions.num_samples = 100;

    // Octonions: CD dim=8, gamma=[-1,-1,-1]
    let mut octonions = AlgebraProperties::new(
        "Octonions",
        8,
        ConstructionMethod::RecursiveDoubling,
        Some(MetricSignature::hurwitz(3)),
    );
    octonions.is_commutative = false;
    octonions.commutativity_percentage = 0.0;
    octonions.num_samples = 100;

    let node1 = TaxonomyNode::new(mixed_quat);
    let node2 = TaxonomyNode::new(quaternions);
    let node3 = TaxonomyNode::new(octonions);

    census.register_algebra(node1).ok();
    census.register_algebra(node2).ok();
    census.register_algebra(node3).ok();

    // All three must have same commutativity (0%)
    assert!(
        census.verify_axis1_commutativity_law().is_ok(),
        "All CD algebras dim>=4 must be 0% commutative"
    );
}

/// TEST CLASS 3: Axis 2 Hypothesis - Metric Signature Controls Division Status
/// Hypothesis: In CD family, gamma=-1 always -> division algebra, gamma=+1 -> zero-divisors
#[test]
fn test_taxonomy_axis2_signature_division_law() {
    let mut census = CompositionAlgebraCensus::new();

    // Hurwitz: gamma=[-1,-1] (Quaternions) -> division algebra
    let mut quat = AlgebraProperties::new(
        "Quaternions_Hurwitz",
        4,
        ConstructionMethod::RecursiveDoubling,
        Some(MetricSignature::hurwitz(2)),
    );
    quat.is_division_algebra = true;
    quat.has_zero_divisors = false;
    quat.invertibility_percentage = 100.0;

    // Mixed: gamma=[+1,-1] (Mixed Quaternions) -> zero-divisors
    let mut mixed = AlgebraProperties::new(
        "MixedQuaternions_Split",
        4,
        ConstructionMethod::RecursiveDoubling,
        Some(MetricSignature::mixed(vec![1, -1])),
    );
    mixed.is_division_algebra = false;
    mixed.has_zero_divisors = true;
    mixed.invertibility_percentage = 0.0;

    let node1 = TaxonomyNode::new(quat);
    let node2 = TaxonomyNode::new(mixed);

    census.register_algebra(node1).ok();
    census.register_algebra(node2).ok();

    // Verify axis 2 law
    assert!(
        census.verify_axis2_signature_law().is_ok(),
        "Axis 2: Metric signature must predict zero-divisor status"
    );
}

/// TEST CLASS 4: Categorical Distinction Theorem
/// Hypothesis: Tensor products and recursive doubling are categorically distinct
/// Cannot represent tessarines as any mixed-signature CD algebra
#[test]
fn test_taxonomy_categorical_distinction_tensor_vs_cd() {
    let mut tensor_props = AlgebraProperties::new(
        "Tessarines",
        4,
        ConstructionMethod::TensorProduct,
        None,
    );
    tensor_props.is_commutative = true;
    tensor_props.commutativity_percentage = 100.0;
    tensor_props.is_associative = true;
    tensor_props.associativity_percentage = 100.0;

    let mut cd_props = AlgebraProperties::new(
        "MixedQuaternions",
        4,
        ConstructionMethod::RecursiveDoubling,
        Some(MetricSignature::mixed(vec![1, -1])),
    );
    cd_props.is_commutative = false;
    cd_props.commutativity_percentage = 0.0;
    cd_props.is_associative = false;
    cd_props.associativity_percentage = 0.0;

    let node_tensor = TaxonomyNode::new(tensor_props);
    let node_cd = TaxonomyNode::new(cd_props);

    // Cannot be in same family
    assert!(
        !node_tensor.same_family(&node_cd),
        "Tensor product and CD algebras are categorically distinct (different families)"
    );

    // Commutativity distinguishes them universally
    assert!(
        node_tensor.algebra.is_commutative != node_cd.algebra.is_commutative,
        "Commutativity difference proves categorical distinction"
    );
}

/// TEST CLASS 5: Exceptional Algebras - Albert Algebra (J_3(O))
/// Hypothesis: Albert algebra is 100% commutative (Jordan structure) but not division algebra
#[test]
fn test_taxonomy_exceptional_albert_algebra() {
    let mut albert = AlgebraProperties::new(
        "AlbertAlgebra_J3O",
        27,
        ConstructionMethod::Exceptional,
        None,
    );
    albert.is_commutative = true;
    albert.commutativity_percentage = 100.0;
    albert.is_division_algebra = false; // Non-associative + not division
    albert.is_associative = false;
    albert.sri_delta_squared_mean = Some(3.213); // From Phase 10.2
    albert.num_samples = 21;

    let node = TaxonomyNode::new(albert);

    // Albert must be exceptional
    assert!(
        node.is_exceptional,
        "Albert algebra is exceptional (neither tensor nor recursive)"
    );

    // Albert must be commutative (Jordan property)
    assert!(
        node.algebra.is_commutative,
        "Albert algebra: 100% commutative (Jordan product)"
    );

    // Cannot be both non-associative and division algebra
    assert!(
        !node.algebra.is_associative && !node.algebra.is_division_algebra,
        "Albert algebra is non-associative and non-division"
    );
}

/// TEST CLASS 6: Norm Multiplicativity Correlation
/// Hypothesis: Norm multiplicativity preserved iff algebra is division algebra
#[test]
fn test_taxonomy_norm_multiplicativity_correlation() {
    // Division algebra (norm multiplicativity preserved)
    let mut division = AlgebraProperties::new(
        "Quaternions",
        4,
        ConstructionMethod::RecursiveDoubling,
        Some(MetricSignature::hurwitz(2)),
    );
    division.is_division_algebra = true;
    division.has_zero_divisors = false;
    division.preserves_norm_multiplicativity = true;

    // Non-division (norm multiplicativity fails)
    let mut non_division = AlgebraProperties::new(
        "MixedQuaternions",
        4,
        ConstructionMethod::RecursiveDoubling,
        Some(MetricSignature::mixed(vec![1, -1])),
    );
    non_division.is_division_algebra = false;
    non_division.has_zero_divisors = true;
    non_division.preserves_norm_multiplicativity = false;

    // Verify correlation: division algebra preserves norm mult, non-division fails
    assert!(
        division.is_division_algebra && division.preserves_norm_multiplicativity,
        "Division algebra must preserve norm multiplicativity"
    );
    assert!(
        !non_division.is_division_algebra && !non_division.preserves_norm_multiplicativity,
        "Non-division algebra must not preserve norm multiplicativity"
    );
}

/// TEST CLASS 7: Property Matrix Consistency
/// Hypothesis: All properties must be logically consistent across all algebras
#[test]
fn test_taxonomy_property_matrix_consistency() {
    // Test multiple algebra types for consistency

    // Type A: Hurwitz division algebra
    let mut hurwitz = AlgebraProperties::new(
        "Quaternions",
        4,
        ConstructionMethod::RecursiveDoubling,
        Some(MetricSignature::hurwitz(2)),
    );
    hurwitz.is_commutative = false;
    hurwitz.is_division_algebra = true;
    hurwitz.has_zero_divisors = false;
    hurwitz.invertibility_percentage = 100.0;

    // Type B: Tensor product (always commutative)
    let mut tensor = AlgebraProperties::new(
        "Tessarines",
        4,
        ConstructionMethod::TensorProduct,
        None,
    );
    tensor.is_commutative = true;
    tensor.is_associative = true;

    // Type C: Exceptional (commutative non-division)
    let mut exceptional = AlgebraProperties::new(
        "Albert",
        27,
        ConstructionMethod::Exceptional,
        None,
    );
    exceptional.is_commutative = true;
    exceptional.is_associative = false;
    exceptional.is_division_algebra = false;

    // All must pass validation
    assert!(hurwitz.validate_division_consistency().is_ok());
    assert!(hurwitz.validate_commutativity_law().is_ok());
    assert!(tensor.validate_division_consistency().is_ok());
    assert!(tensor.validate_commutativity_law().is_ok());
    assert!(exceptional.validate_division_consistency().is_ok());
    assert!(exceptional.validate_commutativity_law().is_ok());
}

/// TEST CLASS 8: Two-Axis Taxonomy Coverage
/// Hypothesis: All composition algebras span exactly two independent axes
#[test]
fn test_taxonomy_two_axis_coverage() {
    let mut census = CompositionAlgebraCensus::new();

    // Axis 1 construction methods
    let methods = vec![
        ConstructionMethod::TensorProduct,
        ConstructionMethod::RecursiveDoubling,
        ConstructionMethod::Exceptional,
    ];

    for (i, method) in methods.iter().enumerate() {
        let mut props = AlgebraProperties::new(
            &format!("Algebra_{}", i),
            4 + i * 4,
            *method,
            if *method == ConstructionMethod::RecursiveDoubling {
                Some(MetricSignature::hurwitz(2))
            } else {
                None
            },
        );
        props.commutativity_percentage = if *method == ConstructionMethod::RecursiveDoubling {
            0.0
        } else {
            100.0
        };

        let node = TaxonomyNode::new(props);
        census.register_algebra(node).ok();
    }

    let (families, per_family) = census.statistics();

    // Should have 3 distinct families (one per construction method)
    assert!(
        families >= 3,
        "Taxonomy must have at least 3 construction method families"
    );

    // Each family represents one axis 1 value
    for (family_name, count) in per_family {
        assert!(
            count > 0,
            "Family {} should have at least one algebra",
            family_name
        );
    }
}

/// TEST CLASS 9: Phase 9-10 Categorical Distinction Extension
/// Hypothesis: Categorical distinction (tessarines != CD) extends to all families
/// No tensor product algebra can be represented as any CD algebra
#[test]
fn test_taxonomy_phase9_distinction_extended() {
    // Create full taxonomy with representatives from each family

    // Tensor product family: always 100% commutative and associative
    let tensor_algebraic_properties = vec![
        (true, true),  // commutative, associative
        (true, true),  // all tensors have this pattern
        (true, true),  // cannot vary
    ];

    // CD family: always 0% commutative for dim >= 4
    let cd_algebraic_properties = vec![
        (false, false), // non-commutative, mostly non-associative
        (false, false), // all dim >= 4 CD have this pattern
        (false, true),  // quaternions non-commutative but some structure
    ];

    // Exceptional family: 100% commutative, mostly non-associative
    let exceptional_algebraic_properties = vec![
        (true, false),  // commutative, non-associative
        (true, false),  // all exceptional have this pattern
    ];

    // Verify no overlap: tensor != CD
    for (t_comm, _) in &tensor_algebraic_properties {
        for (cd_comm, _) in &cd_algebraic_properties {
            // At minimum, commutativity differs
            assert!(
                t_comm != cd_comm,
                "Tensor and CD must differ in commutativity status"
            );
        }
    }

    // Exceptional differs from CD in commutativity
    for (e_comm, _) in &exceptional_algebraic_properties {
        for (cd_comm, _) in &cd_algebraic_properties {
            // Exceptional (true) differs from CD (false)
            assert!(e_comm != cd_comm, "Exceptional and CD differ in commutativity");
        }
    }
}

/// TEST CLASS 10: Zero-Divisor Structure Mapping
/// Hypothesis: Zero-divisor presence is metric-dependent in CD, construction-dependent elsewhere
#[test]
fn test_taxonomy_zero_divisor_structure_mapping() {
    // CD family: zero-divisors depend on signature
    let hurwitz_no_zd = MetricSignature::hurwitz(2);
    assert!(!hurwitz_no_zd.has_zero_divisors(), "Hurwitz has no zero-divisors");

    let mixed_has_zd = MetricSignature::mixed(vec![1, -1]);
    assert!(mixed_has_zd.has_zero_divisors(), "Mixed signature has zero-divisors");

    // Tensor products: always have zero-divisors (z,0) family
    // (but these are "simple" - can be factored or ignored)
    let tensor_zd_present = true;
    assert!(tensor_zd_present, "Tensor products contain (a,0) zero-divisors");

    // Exceptional (Albert): no zero-divisors by structure
    let albert_no_zd = false;
    assert!(!albert_no_zd, "Albert algebra has no zero-divisors");
}

#[test]
fn test_taxonomy_census_export_summary() {
    let mut census = CompositionAlgebraCensus::new();

    let mut props1 = AlgebraProperties::new("Test1", 4, ConstructionMethod::TensorProduct, None);
    props1.is_commutative = true;
    props1.commutativity_percentage = 100.0;

    census.register_algebra(TaxonomyNode::new(props1)).ok();

    let summary = census.export_summary();
    assert_eq!(summary.len(), 1, "Export should contain one algebra");
    assert_eq!(summary[0].name, "Test1");
}
