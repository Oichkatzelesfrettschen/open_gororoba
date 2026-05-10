//! Kac-Moody Algebras: Infinite-Dimensional Generalizations of Lie Algebras.
//!
//! This module implements generalized Cartan matrices (GCMs) and their classification
//! into finite, affine, and indefinite (hyperbolic) types. Kac-Moody algebras are
//! fundamental to string theory, conformal field theory, and M-theory.
//!
//! # Classification
//!
//! A generalized Cartan matrix A is:
//! - **Finite type**: A is positive definite (classical Lie algebras A_n, D_n, E_6, E_7, E_8)
//! - **Affine type**: A is positive semi-definite with corank 1 (loop algebras, E_9 = E_8^{(1)})
//! - **Indefinite type**: A has signature (n-1, 1) or worse (E_10, E_11, etc.)
//!   - **Hyperbolic**: rank n indefinite with every proper connected subdiagram finite or affine
//!   - **Lorentzian**: signature (n-1, 1)
//!
//! # E-series Extensions
//!
//! The E-series extends beyond E8:
//! - E_9 = E_8^{(1)}: Affine extension, important in string theory
//! - E_10: Hyperbolic, conjectured to encode M-theory symmetries
//! - E_11: Even larger, proposed as hidden symmetry of supergravity
//!
//! # Literature
//!
//! - Kac, V. G. (1990). Infinite-Dimensional Lie Algebras (3rd ed.). Cambridge.
//! - Damour, T., Henneaux, M., & Nicolai, H. (2002). E10 and a 'small tension expansion' of M-theory. PRL 89.
//! - West, P. (2001). E11 and M Theory. Class. Quantum Grav. 18, 4443.
//! - Nicolai, H. & Samtleben, H. (2005). On K(E9). Q.J.Math. 56, 403-423.

pub mod cartans;
pub mod e_series;
pub mod roots;
pub mod types;
pub use cartans::{a_n_cartan, d_n_cartan, e8_cartan, e9_cartan, e10_cartan, e11_cartan};
pub use e_series::{E9RootSystem, E10RootSystem, E11RootSystem, ESeriesRootSystem};
pub use roots::{KacMoodyRoot, KacMoodyRootSystem, RootType};
pub use types::{
    CartanEntry, DynkinDiagram, DynkinEdge, DynkinNode, GeneralizedCartanMatrix, KacMoodyType,
    LieAlgebraType, WeylGroupInfo,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_e8_cartan_finite() {
        let e8 = e8_cartan();
        assert_eq!(e8.rank(), 8);
        assert!(e8.is_simply_laced());
        assert_eq!(e8.determinant(), 1);
        assert_eq!(e8.classify(), KacMoodyType::Finite);
        assert_eq!(e8.identify_algebra_type(), Some(LieAlgebraType::E8));
    }

    #[test]
    fn test_e9_cartan_affine() {
        let e9 = e9_cartan();
        assert_eq!(e9.rank(), 9);
        assert!(e9.is_simply_laced());
        assert_eq!(e9.determinant(), 0);
        assert_eq!(e9.classify(), KacMoodyType::Affine);
        assert_eq!(e9.identify_algebra_type(), Some(LieAlgebraType::E9));
    }

    #[test]
    fn test_e10_cartan_hyperbolic() {
        let e10 = e10_cartan();
        assert_eq!(e10.rank(), 10);
        assert!(e10.is_simply_laced());
        // E10 has determinant -1 (Lorentzian signature)
        assert!(e10.determinant() < 0);
        let classification = e10.classify();
        assert!(
            classification == KacMoodyType::Hyperbolic
                || classification == KacMoodyType::Lorentzian
        );
    }

    #[test]
    fn test_e11_cartan() {
        let e11 = e11_cartan();
        assert_eq!(e11.rank(), 11);
        assert!(e11.is_simply_laced());
        // E11 is indefinite
        let classification = e11.classify();
        assert!(classification != KacMoodyType::Finite);
        assert!(classification != KacMoodyType::Affine);
    }

    #[test]
    fn test_a_n_series() {
        for n in 2..=5 {
            let a_n = a_n_cartan(n);
            assert_eq!(a_n.rank(), n);
            assert!(a_n.is_simply_laced());
            assert_eq!(a_n.determinant(), (n + 1) as i64);
            assert_eq!(a_n.classify(), KacMoodyType::Finite);
        }
    }

    #[test]
    fn test_d_n_series() {
        for n in 4..=6 {
            let d_n = d_n_cartan(n);
            assert_eq!(d_n.rank(), n);
            assert!(d_n.is_simply_laced());
            assert_eq!(d_n.determinant(), 4);
            assert_eq!(d_n.classify(), KacMoodyType::Finite);
        }
    }

    #[test]
    fn test_weyl_group_orders() {
        let e8 = e8_cartan();
        let weyl = e8.weyl_group_info();
        assert!(weyl.is_finite);
        assert_eq!(weyl.order, Some(696729600));

        let e9 = e9_cartan();
        let weyl9 = e9.weyl_group_info();
        assert!(!weyl9.is_finite);
        assert_eq!(weyl9.order, None);
    }

    #[test]
    fn test_dynkin_diagram_e8() {
        let e8 = e8_cartan();
        let diagram = e8.dynkin_diagram();

        assert_eq!(diagram.nodes.len(), 8);
        assert_eq!(diagram.edges.len(), 7); // E8 has 7 edges (tree structure)

        // All edges should be single bonds (simply-laced)
        for edge in &diagram.edges {
            assert_eq!(edge.multiplicity, 1);
            assert!(edge.arrow_to_shorter.is_none());
        }
    }

    #[test]
    fn test_root_system_reflection() {
        let a2 = a_n_cartan(2);
        let root_sys = KacMoodyRootSystem::from_gcm(a2);

        let weight = vec![1.0, 0.0];
        let reflected = root_sys.simple_reflection(&weight, 0);

        // s_1(e_1) should give something different
        assert!(
            (reflected[0] - weight[0]).abs() > 1e-10 || (reflected[1] - weight[1]).abs() > 1e-10
        );
    }

    #[test]
    fn test_e_series_hierarchy() {
        // E8 -> E9 -> E10 -> E11 form a hierarchy
        let e8 = e8_cartan();
        let e9 = e9_cartan();
        let e10 = e10_cartan();
        let e11 = e11_cartan();

        // Check ranks increase
        assert_eq!(e8.rank(), 8);
        assert_eq!(e9.rank(), 9);
        assert_eq!(e10.rank(), 10);
        assert_eq!(e11.rank(), 11);

        // Check E8 is finite, E9 affine, E10+ indefinite
        assert_eq!(e8.classify(), KacMoodyType::Finite);
        assert_eq!(e9.classify(), KacMoodyType::Affine);
        assert!(e10.classify() != KacMoodyType::Finite);
        assert!(e10.classify() != KacMoodyType::Affine);
    }

    // === E9/E10/E11 Root System Tests ===

    #[test]
    fn test_e9_root_system_creation() {
        let e9 = E9RootSystem::new();

        // Should have 8 E8 simple roots
        assert_eq!(e9.e8_simple_roots.len(), 8);

        // Delta should be at level 1
        assert_eq!(e9.delta.level, 1);
        assert_eq!(e9.delta.root_type, RootType::Null);

        // Cartan matrix should be 9x9 with determinant 0
        assert_eq!(e9.cartan.rank(), 9);
        assert_eq!(e9.cartan.determinant(), 0);
    }

    #[test]
    fn test_e9_imaginary_root_multiplicity() {
        let e9 = E9RootSystem::new();

        // Level 0 has no imaginary roots
        assert_eq!(e9.imaginary_root_multiplicity(0), 0);

        // Other levels have multiplicity 8 (E8 Cartan dimension)
        assert_eq!(e9.imaginary_root_multiplicity(1), 8);
        assert_eq!(e9.imaginary_root_multiplicity(-1), 8);
        assert_eq!(e9.imaginary_root_multiplicity(5), 8);
    }

    #[test]
    fn test_e9_real_roots_generation() {
        let e9 = E9RootSystem::new();
        let roots = e9.real_roots_up_to_level(2);

        // Should have roots at levels -2, -1, 0, 1, 2
        // Level 0: 16 (8 simple + 8 negative)
        // Levels +/-1: 16 each
        // Levels +/-2: 16 each
        assert!(roots.len() >= 16); // At minimum level 0

        // Check some roots are at different levels
        let levels: std::collections::HashSet<i32> = roots.iter().map(|r| r.level).collect();
        assert!(levels.contains(&0));
    }

    #[test]
    fn test_e10_root_system_creation() {
        let e10 = E10RootSystem::new();

        // Cartan matrix should be 10x10
        assert_eq!(e10.cartan.rank(), 10);

        // Signature should be (9, 1)
        let (pos, neg) = E10RootSystem::signature();
        assert_eq!(pos, 9);
        assert_eq!(neg, 1);
    }

    #[test]
    fn test_e10_simple_roots() {
        let e10 = E10RootSystem::new();
        let simple_roots = e10.simple_roots();

        // Should have 10 simple roots
        assert_eq!(simple_roots.len(), 10);

        // Last root should have Lorentzian coordinates
        let hyperbolic = &simple_roots[9];
        assert!(!hyperbolic.lorentz_coords.is_empty());
    }

    #[test]
    fn test_e10_inner_product_lorentzian() {
        let e10 = E10RootSystem::new();

        // Create a spacelike root
        let spacelike = KacMoodyRoot::real(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(e10.causal_type(&spacelike), "spacelike");

        // Create a timelike root (Lorentzian direction dominant)
        let timelike = KacMoodyRoot::lorentzian(vec![0.0; 8], 0, vec![2.0]);
        assert_eq!(e10.causal_type(&timelike), "timelike");

        // Create a null root
        let null =
            KacMoodyRoot::lorentzian(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0, vec![1.0]);
        assert_eq!(e10.causal_type(&null), "null");
    }

    #[test]
    fn test_e11_root_system_creation() {
        let e11 = E11RootSystem::new();

        // Cartan matrix should be 11x11
        assert_eq!(e11.cartan.rank(), 11);

        // Should have physics connections
        let connection = E11RootSystem::supergravity_connection();
        assert!(connection.contains("11D supergravity"));
    }

    #[test]
    fn test_e11_mtheory_decomposition() {
        let decomp = E11RootSystem::mtheory_level_decomposition();

        // Should have at least the known low-level representations
        assert!(decomp.len() >= 3);

        // Level 0 should be Lorentz group
        assert_eq!(decomp[0].0, 0);
        assert!(decomp[0].1.contains("SO"));

        // Level 1 should be 3-form (M2-brane)
        assert_eq!(decomp[1].0, 1);
        assert!(decomp[1].2.contains("M2"));
    }

    #[test]
    fn test_kac_moody_root_arithmetic() {
        let a = KacMoodyRoot::affine(vec![1.0, 0.0, 0.0], 1);
        let b = KacMoodyRoot::affine(vec![0.0, 1.0, 0.0], 2);

        let sum = a.add(&b);

        assert_eq!(sum.finite_part, vec![1.0, 1.0, 0.0]);
        assert_eq!(sum.level, 3);
    }

    #[test]
    fn test_eseries_unified_interface() {
        let e9_sys = ESeriesRootSystem::E9(E9RootSystem::new());
        let e10_sys = ESeriesRootSystem::E10(E10RootSystem::new());
        let e11_sys = ESeriesRootSystem::E11(E11RootSystem::new());

        // Check ranks
        assert_eq!(e9_sys.rank(), 9);
        assert_eq!(e10_sys.rank(), 10);
        assert_eq!(e11_sys.rank(), 11);

        // Check classifications
        assert_eq!(e9_sys.classification(), KacMoodyType::Affine);
        assert_eq!(e10_sys.classification(), KacMoodyType::Hyperbolic);
        assert_eq!(e11_sys.classification(), KacMoodyType::Indefinite);

        // Only E8 has finite Weyl group
        assert!(!e9_sys.has_finite_weyl_group());
        assert!(!e10_sys.has_finite_weyl_group());
        assert!(!e11_sys.has_finite_weyl_group());
    }

    #[test]
    fn test_physics_applications() {
        let e10_sys = ESeriesRootSystem::E10(E10RootSystem::new());
        let apps = e10_sys.physics_applications();

        assert!(!apps.is_empty());
        assert!(apps.iter().any(|s| s.contains("M-theory")));
    }

    /// Verify that e8_cartan() is exactly the Gram matrix of the E8 simple root vectors.
    ///
    /// This is the canonical consistency check: C[i][j] must equal <alpha_i, alpha_j>
    /// for the root vectors defined in E9RootSystem::new().
    #[test]
    fn test_e8_cartan_matches_root_vector_gram_matrix() {
        let e8_cartan = e8_cartan();
        let e9 = E9RootSystem::new();
        let roots = &e9.e8_simple_roots;

        assert_eq!(roots.len(), 8);
        for i in 0..8 {
            for j in 0..8 {
                let gram: f64 = roots[i]
                    .finite_part
                    .iter()
                    .zip(roots[j].finite_part.iter())
                    .map(|(a, b)| a * b)
                    .sum();
                let cartan = e8_cartan.get(i, j) as f64;
                assert!(
                    (gram - cartan).abs() < 1e-10,
                    "Gram[{i},{j}] = {gram} but Cartan[{i},{j}] = {cartan}"
                );
            }
        }
    }

    /// Verify the E10 Cartan matrix matches the inner products of E10 simple roots.
    #[test]
    fn test_e10_cartan_matches_simple_root_inner_products() {
        let e10 = E10RootSystem::new();
        let roots = e10.simple_roots();
        let cartan = e10_cartan();

        assert_eq!(roots.len(), 10);
        for i in 0..10 {
            for j in 0..10 {
                let ip = e10.inner_product(&roots[i], &roots[j]);
                let expected = cartan.get(i, j) as f64;
                assert!(
                    (ip - expected).abs() < 1e-8,
                    "E10 inner product <alpha_{i}, alpha_{j}> = {ip} but Cartan[{i},{j}] = {expected}"
                );
            }
        }
    }

    /// Verify E8 branching node is at index 4 (degree 3 in the Dynkin diagram).
    #[test]
    fn test_e8_branch_at_node_4() {
        let e8 = e8_cartan();
        for node in 0..8 {
            let degree: usize = (0..8)
                .filter(|&j| j != node && e8.get(node, j) == -1)
                .count();
            if node == 4 {
                assert_eq!(degree, 3, "Node 4 should be the branch (degree 3)");
            } else {
                assert!(degree <= 2, "Node {node} has unexpected degree {degree}");
            }
        }
    }

    /// Verify E8 highest root theta has the correct properties.
    #[test]
    fn test_e8_highest_root_theta() {
        let e9 = E9RootSystem::new();
        let roots = &e9.e8_simple_roots;

        // Coxeter labels for our numbering (branch at node 4)
        let labels: [f64; 8] = [2.0, 3.0, 4.0, 5.0, 6.0, 3.0, 4.0, 2.0];
        let mut theta = [0.0f64; 8];
        for (i, &label) in labels.iter().enumerate() {
            for (t, &r) in theta.iter_mut().zip(roots[i].finite_part.iter()) {
                *t += label * r;
            }
        }

        // theta should be (1, 0, 0, 0, 0, 0, 0, -1)
        assert!((theta[0] - 1.0).abs() < 1e-10, "theta[0] = {}", theta[0]);
        for (k, &val) in theta.iter().enumerate().take(7).skip(1) {
            assert!(val.abs() < 1e-10, "theta[{k}] = {val}");
        }
        assert!((theta[7] + 1.0).abs() < 1e-10, "theta[7] = {}", theta[7]);

        // |theta|^2 = 2
        let norm_sq: f64 = theta.iter().map(|x| x * x).sum();
        assert!((norm_sq - 2.0).abs() < 1e-10, "|theta|^2 = {norm_sq}");

        // <theta, alpha_0> = 1, <theta, alpha_i> = 0 for i > 0
        for (i, root) in roots.iter().enumerate() {
            let ip: f64 = theta
                .iter()
                .zip(root.finite_part.iter())
                .map(|(a, b)| a * b)
                .sum();
            if i == 0 {
                assert!((ip - 1.0).abs() < 1e-10, "<theta, alpha_0> = {ip}");
            } else {
                assert!(ip.abs() < 1e-10, "<theta, alpha_{i}> = {ip}");
            }
        }
    }
}
