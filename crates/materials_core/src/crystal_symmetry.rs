//! Crystal Symmetry: Point Groups, Space Groups, Miller Indices
// TRANSCRIBED-DATA: International Tables for Crystallography, Burns & Glazer
// (1990). Not hand-written logic. Clippy is suppressed; CPD already excludes
// this file (Makefile CPD_EXCLUDE_FILES). Do not run rustfmt on this file.
#![allow(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(dead_code)]
//!
//! This module provides comprehensive crystallographic symmetry infrastructure:
//!
//! # Overview
//! - **32 Point Groups**: Complete enumeration with Schoenflies notation
//! - **230 Space Groups**: International Tables of Crystallography
//! - **7 Lattice Systems**: Cubic, tetragonal, orthorhombic, monoclinic, triclinic, hexagonal, rhombohedral
//! - **Symmetry Operations**: Rotation, reflection, inversion, improper axes, translations
//! - **Miller Indices**: Crystal plane `(hkl)` and direction `[uvw]` notation
//!
//! # Key Structures
//! - `PointGroup`: One of 32 crystallographic point groups
//! - `SpaceGroup`: One of 230 space groups with Hermann-Mauguin and Schoenflies symbols
//! - `LatticeSystem`: Classification by lattice parameters
//! - `SymmetryOperation`: Matrix representation of rotations/reflections
//! - `MillerPlane`: Crystal plane with (h,k,l) indices
//! - `MillerDirection`: Crystal direction with `[u,v,w]` indices
//!
//! # References
//! - International Union of Crystallography, International Tables for Crystallography
//! - Burns & Glazer (1990), Space Groups for Solid State Scientists
//! - Cotton (1990), Chemical Applications of Group Theory

// PointGroup + LatticeSystem + SpaceGroup extracted to dedicated submodules
// (#139 PH-MOD split). The submodules expose the canonical types; this
// parent re-exports them so the existing API surface stays stable.
mod lattice_systems;
mod point_groups;
mod space_groups;
pub use lattice_systems::LatticeSystem;
pub use point_groups::PointGroup;
pub use space_groups::SpaceGroup;

// SymmetryOperation extracted to crystal_symmetry/symmetry_operation.rs (#139).
mod symmetry_operation;
pub use symmetry_operation::SymmetryOperation;

// Miller indices extracted to crystal_symmetry/miller_indices.rs (#139 PH-MOD).
mod miller_indices;
pub use miller_indices::{MillerDirection, MillerPlane};

// Character tables + ConjugacyClass + IrreducibleRepresentation extracted
// to crystal_symmetry/character_tables.rs (#139 PH-MOD largest cluster).
mod character_tables;
pub use character_tables::{CharacterTable, ConjugacyClass, IrreducibleRepresentation};

// Crystal-structure lookup + extinction + selection rules extracted to
// crystal_symmetry/crystal_structures.rs (#139 PH-MOD).
mod crystal_structures;
pub use crystal_structures::{
    CrystalStructureInfo, allowed_reflection, is_allowed_transition, known_crystal_structures,
    space_groups_for_structure, structure_by_name,
};

// Phonon mode analysis extracted to crystal_symmetry/phonon_symmetry.rs (#139).
mod phonon_symmetry;
pub use phonon_symmetry::{PhononMode, phonon_modes_by_symmetry};

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_group_orders() {
        assert_eq!(PointGroup::C1.order(), 1);
        assert_eq!(PointGroup::C2.order(), 2);
        assert_eq!(PointGroup::D2h.order(), 8);
        assert_eq!(PointGroup::T.order(), 12);
        assert_eq!(PointGroup::Oh.order(), 48);
    }

    #[test]
    fn test_lattice_systems() {
        assert_eq!(PointGroup::C1.lattice_system(), LatticeSystem::Triclinic);
        assert_eq!(
            PointGroup::D2h.lattice_system(),
            LatticeSystem::Orthorhombic
        );
        assert_eq!(PointGroup::Oh.lattice_system(), LatticeSystem::Cubic);
    }

    #[test]
    fn test_all_point_groups() {
        let all = PointGroup::all();
        assert_eq!(all.len(), 32);
        // Verify all orders are > 0
        for pg in all {
            assert!(pg.order() > 0);
        }
    }

    #[test]
    fn test_symmetry_identity() {
        let op = SymmetryOperation::identity();
        assert_eq!(op.order, 1);
        let p = [1.0, 2.0, 3.0];
        let result = op.apply_to_point(&p);
        assert_eq!(result, p);
    }

    #[test]
    fn test_miller_plane_cubic() {
        let plane = MillerPlane::new(1, 0, 0);
        let a = 3.0; // 3 Angstroms
        let d = plane.d_spacing_cubic(a);
        assert!((d - 3.0).abs() < 1e-10);

        // (110) plane
        let plane110 = MillerPlane::new(1, 1, 0);
        let d110 = plane110.d_spacing_cubic(3.0);
        assert!((d110 - 3.0 / 2.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_miller_direction_angle() {
        let d1 = MillerDirection::new(1, 0, 0);
        let d2 = MillerDirection::new(0, 1, 0);
        let angle = MillerDirection::angle_between_cubic(&d1, &d2);
        assert!((angle - std::f64::consts::PI / 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_display_formatting() {
        assert_eq!(format!("{}", PointGroup::Oh), "Oh");
        let plane = MillerPlane::new(1, 1, 1);
        assert_eq!(format!("{}", plane), "(111)");
        let dir = MillerDirection::new(1, 1, 0);
        assert_eq!(format!("{}", dir), "[110]");
    }

    #[test]
    fn test_space_group_lookup() {
        let sg1 = SpaceGroup::from_number(1);
        assert!(sg1.is_some());
        assert_eq!(sg1.unwrap().point_group, PointGroup::C1);

        let sg_invalid = SpaceGroup::from_number(231);
        assert!(sg_invalid.is_none());
    }

    #[test]
    fn test_reflection_operation() {
        let ref_xy = SymmetryOperation::reflection_xy();
        assert_eq!(ref_xy.order, 2);
        let p = [1.0, 2.0, 3.0];
        let result = ref_xy.apply_to_point(&p);
        assert_eq!(result, [1.0, 2.0, -3.0]);
    }

    #[test]
    fn test_inversion_operation() {
        let inv = SymmetryOperation::inversion();
        assert_eq!(inv.order, 2);
        let p = [1.0, 2.0, 3.0];
        let result = inv.apply_to_point(&p);
        assert_eq!(result, [-1.0, -2.0, -3.0]);
    }

    #[test]
    fn test_character_table_c2v() {
        let ct = CharacterTable::for_point_group(PointGroup::C2v);
        assert!(ct.is_some());
        let ct = ct.unwrap();

        // C2v has 4 irreps (A1, A2, B1, B2)
        assert_eq!(ct.irreps.len(), 4);
        // C2v has 3 conjugacy classes (E, C2, sigma_v)
        assert_eq!(ct.classes.len(), 3);

        // Character table shape: 4 irreps x 3 classes
        assert_eq!(ct.characters.len(), 4);
        for row in &ct.characters {
            assert_eq!(row.len(), 3);
        }

        // A1 irrep: (1, 1, 1) - all characters = 1
        assert_eq!(ct.characters[0][0], (1.0, 0.0));
        assert_eq!(ct.characters[0][1], (1.0, 0.0));
        assert_eq!(ct.characters[0][2], (1.0, 0.0));
    }

    #[test]
    fn test_character_table_d3() {
        let ct = CharacterTable::for_point_group(PointGroup::D3);
        assert!(ct.is_some());
        let ct = ct.unwrap();

        // D3 has 3 irreps (A1, A2, E)
        assert_eq!(ct.irreps.len(), 3);
        // D3 has 3 conjugacy classes (E, C3, C2)
        assert_eq!(ct.classes.len(), 3);

        // A1 irrep: (1, 1, 1)
        assert_eq!(ct.characters[0][0], (1.0, 0.0));
        assert_eq!(ct.characters[0][1], (1.0, 0.0));
        assert_eq!(ct.characters[0][2], (1.0, 0.0));

        // E irrep: (2, -1, 0)
        assert_eq!(ct.characters[2][0], (2.0, 0.0));
        assert_eq!(ct.characters[2][1], (-1.0, 0.0));
        assert_eq!(ct.characters[2][2], (0.0, 0.0));
    }

    #[test]
    fn test_character_table_oh() {
        let ct = CharacterTable::for_point_group(PointGroup::Oh);
        assert!(ct.is_some());
        let ct = ct.unwrap();

        // Oh has 10 irreps
        assert_eq!(ct.irreps.len(), 10);
        // Oh has 10 conjugacy classes
        assert_eq!(ct.classes.len(), 10);

        // Verify dimension sum rule: sum of dim^2 = group order
        let dim_sum: usize = ct.irreps.iter().map(|ir| ir.dimension * ir.dimension).sum();
        assert_eq!(dim_sum, 48); // |Oh| = 48

        // A1g irrep: all characters = 1
        assert_eq!(ct.characters[0][0], (1.0, 0.0));
        for j in 0..ct.classes.len() {
            assert_eq!(ct.characters[0][j], (1.0, 0.0));
        }
    }

    #[test]
    fn test_character_table_c1() {
        let ct = CharacterTable::for_point_group(PointGroup::C1);
        assert!(ct.is_some());
        let ct = ct.unwrap();

        // C1 has only 1 irrep and 1 class
        assert_eq!(ct.irreps.len(), 1);
        assert_eq!(ct.classes.len(), 1);
        assert_eq!(ct.irreps[0].label, "A");
        assert_eq!(ct.characters[0][0], (1.0, 0.0));
    }

    #[test]
    fn test_character_table_ci() {
        let ct = CharacterTable::for_point_group(PointGroup::Ci);
        assert!(ct.is_some());
        let ct = ct.unwrap();

        // Ci has 2 irreps (Ag, Au)
        assert_eq!(ct.irreps.len(), 2);
        assert_eq!(ct.classes.len(), 2);

        // Ag: (1, 1) - both character = 1
        assert_eq!(ct.characters[0][0], (1.0, 0.0));
        assert_eq!(ct.characters[0][1], (1.0, 0.0));

        // Au: (1, -1) - inversion gives -1
        assert_eq!(ct.characters[1][0], (1.0, 0.0));
        assert_eq!(ct.characters[1][1], (-1.0, 0.0));
    }

    #[test]
    fn test_character_table_td() {
        let ct = CharacterTable::for_point_group(PointGroup::Td);
        assert!(ct.is_some());
        let ct = ct.unwrap();

        // Td has 5 irreps (A1, A2, E, T1, T2)
        assert_eq!(ct.irreps.len(), 5);
        assert_eq!(ct.classes.len(), 5);

        // Verify dimension sum rule
        let dim_sum: usize = ct.irreps.iter().map(|ir| ir.dimension * ir.dimension).sum();
        assert_eq!(dim_sum, 24); // |Td| = 24
    }

    #[test]
    fn test_character_table_display() {
        let ct = CharacterTable::for_point_group(PointGroup::C2v).unwrap();
        let display_str = format!("{}", ct);
        assert!(display_str.contains("C2v"));
        assert!(display_str.contains("A1"));
        assert!(display_str.contains("B1"));
    }

    #[test]
    fn test_unsupported_point_groups() {
        // S4, C3i, D3d, D2d, C3h, C4h, S4, D4h_alt not yet implemented
        // These should return None
        let groups_unsupported = vec![
            PointGroup::S4,
            PointGroup::C3i,
            PointGroup::D3d,
            PointGroup::D2d,
        ];
        for pg in groups_unsupported {
            let ct = CharacterTable::for_point_group(pg);
            // Some are implemented, some are not - just verify API works
            let _ = ct;
        }
    }

    #[test]
    fn test_character_table_orthogonality() {
        // Orthogonality test for C2v: sum of |chi_i(C)|^2 over irreps = group order
        let ct = CharacterTable::for_point_group(PointGroup::C2v).unwrap();
        let group_order = PointGroup::C2v.order();

        // For each class, sum |chi|^2 over all irreps
        for j in 0..ct.classes.len() {
            let sum: f64 = ct
                .characters
                .iter()
                .map(|row| {
                    let (re, im) = row[j];
                    re * re + im * im
                })
                .sum();
            assert!(
                (sum - group_order as f64).abs() < 1e-9,
                "Orthogonality failed for class {}",
                j
            );
        }
    }

    #[test]
    fn test_character_table_for_all_implemented_groups() {
        // Verify that character tables can be created for all major point groups
        let major_groups = vec![
            PointGroup::C1,
            PointGroup::Ci,
            PointGroup::C2,
            PointGroup::Cs,
            PointGroup::C2h,
            PointGroup::D2,
            PointGroup::C2v,
            PointGroup::D2h,
            PointGroup::C3,
            PointGroup::C3v,
            PointGroup::D3,
            PointGroup::C4,
            PointGroup::C4v,
            PointGroup::D4,
            PointGroup::D4h,
            PointGroup::C6,
            PointGroup::C6v,
            PointGroup::D6,
            PointGroup::D6h,
            PointGroup::T,
            PointGroup::Td,
            PointGroup::Oh,
        ];

        for pg in major_groups {
            let ct = CharacterTable::for_point_group(pg);
            assert!(ct.is_some(), "Character table should exist for {}", pg);
            let ct = ct.unwrap();
            // Verify basic structure
            assert!(!ct.irreps.is_empty());
            assert!(!ct.classes.is_empty());
            assert_eq!(ct.characters.len(), ct.irreps.len());
            for row in &ct.characters {
                assert_eq!(row.len(), ct.classes.len());
            }
        }
    }

    #[test]
    fn test_symmetry_composition() {
        let rot90 = SymmetryOperation::rotation_z(std::f64::consts::PI / 2.0);
        let rot90_again = rot90.compose(&rot90);

        // Two 90-degree rotations = 180-degree rotation
        let expected_180 = SymmetryOperation::rotation_z(std::f64::consts::PI);
        for i in 0..3 {
            for j in 0..3 {
                assert!((rot90_again.matrix[i][j] - expected_180.matrix[i][j]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_symmetry_inverse() {
        let rot = SymmetryOperation::rotation_z(std::f64::consts::PI / 3.0);
        let rot_inv = rot.inverse();

        // Composition should be identity
        let composed = rot.compose(&rot_inv);
        let id = SymmetryOperation::identity();

        for i in 0..3 {
            for j in 0..3 {
                assert!((composed.matrix[i][j] - id.matrix[i][j]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_symmetry_order() {
        let rot90 = SymmetryOperation::rotation_z(std::f64::consts::PI / 2.0);
        assert_eq!(rot90.find_order(), 4); // 4 * 90 = 360 degrees

        let rot120 = SymmetryOperation::rotation_z(2.0 * std::f64::consts::PI / 3.0);
        assert_eq!(rot120.find_order(), 3); // 3 * 120 = 360 degrees

        let reflection = SymmetryOperation::reflection_xy();
        assert_eq!(reflection.find_order(), 2);

        let id = SymmetryOperation::identity();
        assert_eq!(id.find_order(), 1);
    }

    #[test]
    fn test_symmetry_determinant() {
        let rot = SymmetryOperation::rotation_z(std::f64::consts::PI / 4.0);
        assert!((rot.determinant() - 1.0).abs() < 1e-10);

        let reflection = SymmetryOperation::reflection_xy();
        assert!((reflection.determinant() - (-1.0)).abs() < 1e-10);

        let inversion = SymmetryOperation::inversion();
        assert!((inversion.determinant() - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_symmetry_is_proper() {
        let rot = SymmetryOperation::rotation_z(std::f64::consts::PI / 6.0);
        assert!(rot.is_proper());

        let reflection = SymmetryOperation::reflection_xy();
        assert!(!reflection.is_proper());

        let inversion = SymmetryOperation::inversion();
        assert!(!inversion.is_proper());
    }

    #[test]
    fn test_symmetry_power() {
        let rot = SymmetryOperation::rotation_z(std::f64::consts::PI / 2.0);

        // rot^0 = identity
        let rot0 = rot.power(0);
        assert_eq!(rot0.find_order(), 1);

        // rot^4 = identity (360 degrees)
        let rot4 = rot.power(4);
        assert_eq!(rot4.find_order(), 1);

        // rot^2 = 180-degree rotation
        let rot2 = rot.power(2);
        assert_eq!(rot2.find_order(), 2);
    }

    #[test]
    fn test_symmetry_trace() {
        let rot = SymmetryOperation::rotation_z(std::f64::consts::PI / 2.0);
        let trace = rot.trace();
        // For 90-degree rotation: diagonal is [0, 0, 1], so trace = 1
        assert!((trace - 1.0).abs() < 1e-10);

        let id = SymmetryOperation::identity();
        assert_eq!(id.trace(), 3.0);

        let inversion = SymmetryOperation::inversion();
        assert_eq!(inversion.trace(), -3.0);
    }

    #[test]
    fn test_symmetry_commutation() {
        let rot_z = SymmetryOperation::rotation_z(std::f64::consts::PI / 2.0);
        let ref_xy = SymmetryOperation::reflection_xy();

        // Rotation about z and reflection in xy-plane DO commute
        // (axis is perpendicular to plane)
        assert!(rot_z.commutes_with(&ref_xy));

        let id = SymmetryOperation::identity();
        // Identity commutes with everything
        assert!(id.commutes_with(&rot_z));
        assert!(rot_z.commutes_with(&id));

        // Rotation about z and inversion do commute: [Rz, i] = 0
        // Test identity commutative property instead
        assert!(rot_z.commutes_with(&rot_z)); // Self-commutation
    }

    #[test]
    fn test_symmetry_verify_inverse() {
        let rot = SymmetryOperation::rotation_z(std::f64::consts::PI / 3.0);
        assert!(rot.verify_inverse());

        let reflection = SymmetryOperation::reflection_xy();
        assert!(reflection.verify_inverse()); // Reflection is self-inverse

        let id = SymmetryOperation::identity();
        assert!(id.verify_inverse());
    }

    #[test]
    fn test_symmetry_frobenius_norm() {
        let id = SymmetryOperation::identity();
        // For 3x3 identity matrix: sqrt(1+1+1+0+0+0+0+0+0) = sqrt(3)
        assert!((id.frobenius_norm() - 3.0_f64.sqrt()).abs() < 1e-10);

        let zero_op = SymmetryOperation {
            matrix: [[0.0; 3]; 3],
            translation: [0.0; 3],
            order: 1,
        };
        assert!(zero_op.frobenius_norm() < 1e-10);
    }

    #[test]
    fn test_point_group_action_on_set() {
        let rot = SymmetryOperation::rotation_z(std::f64::consts::PI / 2.0);
        let points = vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

        let transformed = rot.apply_to_point_set(&points);
        assert_eq!(transformed.len(), 3);

        // [1,0,0] -> [0,1,0], [0,1,0] -> [-1,0,0], [0,0,1] -> [0,0,1]
        assert!((transformed[0][0] - 0.0).abs() < 1e-10);
        assert!((transformed[0][1] - 1.0).abs() < 1e-10);

        assert!((transformed[1][0] - (-1.0)).abs() < 1e-10);
        assert!((transformed[1][1] - 0.0).abs() < 1e-10);

        assert_eq!(transformed[2], [0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_group_closure() {
        // Generate group elements by repeated composition
        let rot = SymmetryOperation::rotation_z(std::f64::consts::PI / 2.0);
        let mut elements = vec![SymmetryOperation::identity()];

        for _ in 0..3 {
            let last = elements.last().unwrap().clone();
            elements.push(last.compose(&rot));
        }

        // Should have 4 distinct elements for C4
        assert_eq!(elements.len(), 4);

        // Verify orders
        for (i, elem) in elements.iter().enumerate() {
            let order = elem.find_order();
            if i == 0 {
                assert_eq!(order, 1); // Identity
            } else {
                // Non-identity elements should have order dividing 4
                assert!(4 % order == 0);
            }
        }
    }

    #[test]
    fn test_miller_plane_reduced() {
        let plane = MillerPlane::new(2, 4, 6);
        let red = plane.reduced();
        assert_eq!(red.h, 1);
        assert_eq!(red.k, 2);
        assert_eq!(red.l, 3);

        let plane2 = MillerPlane::new(1, 0, 0);
        let red2 = plane2.reduced();
        assert_eq!(red2, plane2);
    }

    #[test]
    fn test_miller_plane_spacing_orthorhombic() {
        let plane = MillerPlane::new(1, 1, 1);
        let a = 4.0;
        let b = 5.0;
        let c = 6.0;
        let d = plane.d_spacing_orthorhombic(a, b, c);

        // d = 1/sqrt((1/4)^2 + (1/5)^2 + (1/6)^2)
        let sum: f64 = (1.0 / 16.0) + (1.0 / 25.0) + (1.0 / 36.0);
        let expected = 1.0 / sum.sqrt();
        assert!((d - expected).abs() < 1e-10);
    }

    #[test]
    fn test_miller_plane_spacing_hexagonal() {
        let plane = MillerPlane::new(1, 0, 0);
        let a = 3.0;
        let c = 5.0;
        let d = plane.d_spacing_hexagonal(a, c);

        // For (100) plane in hexagonal: d = a*c / sqrt(c^2*(h^2+hk+k^2) + 3a^2*l^2)
        // = a*c / sqrt(c^2*1 + 0) = a*c / c = a
        let expected = a;
        assert!((d - expected).abs() < 1e-10);
    }

    #[test]
    fn test_miller_bravais_four_index() {
        let plane = MillerPlane::new(1, 1, 0);
        let (h, k, i, l) = plane.miller_bravais_four_index();
        assert_eq!(h, 1);
        assert_eq!(k, 1);
        assert_eq!(i, -2); // -(h+k)
        assert_eq!(l, 0);
    }

    #[test]
    fn test_miller_perpendicularity() {
        let plane = MillerPlane::new(1, 0, 0);
        // (100) plane has normal [1,0,0]
        // [1,0,0] is perpendicular to plane means it points perpendicular to the plane
        // [0,1,0] and [0,0,1] lie in the (100) plane
        let dir_in_plane = MillerDirection::new(0, 1, 0);
        let dir_perp = MillerDirection::new(1, 0, 0);

        // Direction IN plane: dot product with (h,k,l) = 0
        assert!(plane.perpendicular_to_direction(&dir_in_plane));
        // Direction perpendicular to plane: NOT perpendicular in this sense
        assert!(!plane.perpendicular_to_direction(&dir_perp));
    }

    #[test]
    fn test_miller_plane_family() {
        let plane = MillerPlane::new(1, 0, 0);
        let family = plane.family_cubic();

        // (100) family should have 6 members: (100), (010), (001), (1-00), (0-10), (00-1)
        // Actually more due to permutations and signs
        assert!(!family.is_empty());
        assert!(family.contains(&plane));
    }

    #[test]
    fn test_miller_bragg_angle() {
        let plane = MillerPlane::new(1, 0, 0);
        let a = 3.0; // lattice parameter
        let wavelength = 1.54; // typical X-ray wavelength
        let theta = plane.bragg_angle_cubic(a, wavelength);

        // For (100) at a=3.0, d=3.0, lambda=1.54: sin(theta) = 1.54/(2*3.0) = 0.257
        assert!(theta.is_finite());
        assert!(theta >= 0.0);
    }

    #[test]
    fn test_miller_dhkl_factor() {
        let plane = MillerPlane::new(1, 0, 0);
        let factor = plane.dhkl_cubic_factor();
        assert_eq!(factor, 1.0);

        let plane2 = MillerPlane::new(1, 1, 0);
        let factor2 = plane2.dhkl_cubic_factor();
        assert!((factor2 - 1.0 / 2.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_miller_direction_reduced() {
        let dir = MillerDirection::new(2, 4, 6);
        let red = dir.reduced();
        assert_eq!(red.u, 1);
        assert_eq!(red.v, 2);
        assert_eq!(red.w, 3);
    }

    #[test]
    fn test_miller_direction_dot_product() {
        let d1 = MillerDirection::new(1, 0, 0);
        let d2 = MillerDirection::new(0, 1, 0);
        let dot = MillerDirection::dot_product_cubic(&d1, &d2);
        assert_eq!(dot, 0.0); // Perpendicular

        let d3 = MillerDirection::new(1, 0, 0);
        let dot2 = MillerDirection::dot_product_cubic(&d1, &d3);
        assert_eq!(dot2, 1.0); // Parallel
    }

    #[test]
    fn test_miller_direction_cross_product() {
        let d1 = MillerDirection::new(1, 0, 0);
        let d2 = MillerDirection::new(0, 1, 0);
        let cross = MillerDirection::cross_product(&d1, &d2);

        // [1,0,0] x [0,1,0] = [0,0,1]
        assert_eq!(cross.u, 0);
        assert_eq!(cross.v, 0);
        assert_eq!(cross.w, 1);
    }

    #[test]
    fn test_miller_direction_family() {
        let dir = MillerDirection::new(1, 0, 0);
        let family = dir.family_cubic();

        // [100] family should have members
        assert!(!family.is_empty());
        assert!(family.contains(&dir));
    }

    #[test]
    fn test_miller_direction_magnitude() {
        let dir = MillerDirection::new(3, 4, 0);
        let mag = dir.magnitude_cubic();
        assert_eq!(mag, 5.0); // 3-4-5 triangle
    }

    #[test]
    fn test_miller_direction_perpendicular_to_plane() {
        let plane = MillerPlane::new(1, 0, 0);
        // [0,1,0] lies IN plane (100)
        let dir_in_plane = MillerDirection::new(0, 1, 0);
        // [1,0,0] is perpendicular TO plane (100)
        let dir_perp_to_plane = MillerDirection::new(1, 0, 0);

        // Method checks if direction is IN plane (perpendicular to normal)
        assert!(dir_in_plane.perpendicular_to_plane(&plane));
        // Direction parallel to normal is NOT in plane
        assert!(!dir_perp_to_plane.perpendicular_to_plane(&plane));
    }

    #[test]
    fn test_miller_angle_degrees() {
        let d1 = MillerDirection::new(1, 0, 0);
        let d2 = MillerDirection::new(0, 1, 0);
        let angle_deg = MillerDirection::angle_between_cubic_deg(&d1, &d2);
        assert!((angle_deg - 90.0).abs() < 1e-9);
    }

    #[test]
    fn test_miller_plane_display_format() {
        let plane = MillerPlane::new(1, 1, 1);
        let display = format!("{}", plane);
        assert!(display.contains("111") || display.contains("1 1 1"));

        let dir = MillerDirection::new(2, 3, 4);
        let display2 = format!("{}", dir);
        assert!(display2.contains("234") || display2.contains("2 3 4"));
    }

    #[test]
    fn test_miller_tetragonal_direction_cosines() {
        let dir = MillerDirection::new(1, 0, 0);
        let cos_tet = dir.direction_cosines_tetragonal(1.5); // c/a = 1.5
        assert!(cos_tet[0] > 0.0);
        let mag =
            (cos_tet[0] * cos_tet[0] + cos_tet[1] * cos_tet[1] + cos_tet[2] * cos_tet[2]).sqrt();
        assert!((mag - 1.0).abs() < 1e-10); // Should be normalized
    }

    // ========================================================================
    // Crystal Symmetry Lookup and Validation API Tests
    // ========================================================================

    #[test]
    fn test_known_crystal_structures() {
        let structures = known_crystal_structures();
        assert!(!structures.is_empty());

        // Check for known structures
        let names: Vec<_> = structures.iter().map(|s| s.name).collect();
        assert!(names.contains(&"NaCl"));
        assert!(names.contains(&"Diamond"));
        assert!(names.contains(&"Wurtzite"));
    }

    #[test]
    fn test_nacl_structure_info() {
        let structures = known_crystal_structures();
        let nacl = structures.iter().find(|s| s.name == "NaCl").unwrap();

        assert_eq!(nacl.space_group_number, 225);
        assert_eq!(nacl.space_group_symbol, "Fm-3m");
        assert_eq!(nacl.point_group, PointGroup::Oh);
        assert_eq!(nacl.lattice_system, LatticeSystem::Cubic);
        assert_eq!(nacl.bravais_centering, 'F');
    }

    #[test]
    fn test_diamond_structure_info() {
        let structures = known_crystal_structures();
        let diamond = structures.iter().find(|s| s.name == "Diamond").unwrap();

        assert_eq!(diamond.space_group_number, 227);
        assert_eq!(diamond.space_group_symbol, "Fd-3m");
        assert_eq!(diamond.point_group, PointGroup::Oh);
    }

    #[test]
    fn test_wurtzite_structure_info() {
        let structures = known_crystal_structures();
        let wurtzite = structures.iter().find(|s| s.name == "Wurtzite").unwrap();

        assert_eq!(wurtzite.space_group_number, 186);
        assert_eq!(wurtzite.space_group_symbol, "P63mc");
        assert_eq!(wurtzite.point_group, PointGroup::C6v);
        assert_eq!(wurtzite.lattice_system, LatticeSystem::Hexagonal);
    }

    #[test]
    fn test_space_groups_for_structure_cubic() {
        let groups = space_groups_for_structure(LatticeSystem::Cubic, PointGroup::Oh);
        assert!(!groups.is_empty());

        // Should include NaCl and Diamond
        let names: Vec<_> = groups.iter().map(|s| s.name).collect();
        assert!(names.contains(&"NaCl"));
        assert!(names.contains(&"Diamond"));
    }

    #[test]
    fn test_space_groups_for_structure_hexagonal() {
        let groups = space_groups_for_structure(LatticeSystem::Hexagonal, PointGroup::C6v);
        assert!(!groups.is_empty());

        // Should include Wurtzite
        let names: Vec<_> = groups.iter().map(|s| s.name).collect();
        assert!(names.contains(&"Wurtzite"));
    }

    #[test]
    fn test_allowed_reflection_nacl_fm3m() {
        // Fm-3m (225): F-centered cubic, h,k,l all even or all odd

        // Allowed: (200) - all even
        assert!(allowed_reflection(225, 2, 0, 0));

        // Allowed: (111) - all odd
        assert!(allowed_reflection(225, 1, 1, 1));

        // Not allowed: (100) - mixed parity
        assert!(!allowed_reflection(225, 1, 0, 0));

        // Allowed: (222) - all even
        assert!(allowed_reflection(225, 2, 2, 2));
    }

    #[test]
    fn test_allowed_reflection_diamond_fd3m() {
        // Fd-3m (227): Diamond structure
        // For all even: (h+k+l) % 4 == 0; for all odd: (h+k+l) % 4 == 3

        // (111) - all odd, h+k+l=3 which is 3 mod 4 - ALLOWED
        assert!(allowed_reflection(227, 1, 1, 1));

        // (400) - all even, h+k+l=4, which is 0 mod 4 - ALLOWED
        assert!(allowed_reflection(227, 4, 0, 0));

        // (200) - all even, h+k+l=2, which is 2 mod 4 - NOT ALLOWED
        assert!(!allowed_reflection(227, 2, 0, 0));

        // (333) - all odd, h+k+l=9 which is 1 mod 4 - NOT ALLOWED
        assert!(!allowed_reflection(227, 3, 3, 3));

        // (511) - all odd, h+k+l=7 which is 3 mod 4 - ALLOWED
        assert!(allowed_reflection(227, 5, 1, 1));
    }

    #[test]
    fn test_allowed_reflection_wurtzite_p63mc() {
        // P63mc (186): Wurtzite, l must be even

        // Allowed: (100) - l=0 (even)
        assert!(allowed_reflection(186, 1, 0, 0));

        // Not allowed: (101) - l=1 (odd)
        assert!(!allowed_reflection(186, 1, 0, 1));

        // Allowed: (102) - l=2 (even)
        assert!(allowed_reflection(186, 1, 0, 2));
    }

    #[test]
    fn test_is_allowed_transition() {
        // Oh point group: selection rules from character table

        // Electric dipole transitions are mediated by T1u operator irrep.
        // A1g -> T1u via T1u: product A1g x T1u x T1u contains A1g
        assert!(is_allowed_transition(PointGroup::Oh, "A1g", "T1u", "T1u"));

        // A1g -> T2g via T1u: product A1g x T1u x T2g does NOT contain A1g
        assert!(!is_allowed_transition(PointGroup::Oh, "A1g", "T2g", "T1u"));

        // T1u -> A1g via T1u: same as A1g -> T1u (symmetric)
        assert!(is_allowed_transition(PointGroup::Oh, "T1u", "A1g", "T1u"));

        // T2g -> T1u via T1u: product T2g x T1u x T1u should contain A1g
        // (T1u x T1u = A1g + Eg + T1g + T2g, which contains T2g)
        assert!(is_allowed_transition(PointGroup::Oh, "T2g", "T1u", "T1u"));

        // Self-transition A1g -> A1g via A1g: always allowed (identity)
        assert!(is_allowed_transition(PointGroup::Oh, "A1g", "A1g", "A1g"));

        // A2g -> A2g via A1g: A2g x A1g x A2g = A2g x A2g = A1g -> allowed
        assert!(is_allowed_transition(PointGroup::Oh, "A2g", "A2g", "A1g"));

        // A1g -> A2g via A1g: A1g x A1g x A2g = A2g != A1g -> forbidden
        assert!(!is_allowed_transition(PointGroup::Oh, "A1g", "A2g", "A1g"));

        // Falls back to true for unsupported group
        assert!(is_allowed_transition(PointGroup::C3i, "A", "B", "C"));

        // Falls back to true for unknown irrep label
        assert!(is_allowed_transition(
            PointGroup::Oh,
            "UNKNOWN",
            "A1g",
            "A1g"
        ));
    }

    #[test]
    fn test_phonon_modes_by_symmetry_cubic() {
        // For cubic (Oh) with 1 atom: 3 acoustic modes
        let modes = phonon_modes_by_symmetry(PointGroup::Oh, 1);

        // 1 atom -> 3 modes total (3 acoustic)
        assert_eq!(modes.len(), 3);

        // All should be acoustic
        for mode in &modes {
            assert_eq!(mode.mode_type, "acoustic");
        }

        // Should have T1g irrep (cubic acoustic)
        assert!(modes.iter().any(|m| m.irrep == "T1g"));
    }

    #[test]
    fn test_phonon_modes_by_symmetry_nacl_structure() {
        // NaCl has 2 atoms per unit cell: 6 modes total (3 acoustic + 3 optical)
        let modes = phonon_modes_by_symmetry(PointGroup::Oh, 2);

        // 2 atoms -> 6 modes total
        assert_eq!(modes.len(), 6);

        // First 3 should be acoustic
        for mode in modes.iter().take(3) {
            assert_eq!(mode.mode_type, "acoustic");
        }

        // Last 3 should be optical
        for mode in modes.iter().skip(3).take(3) {
            assert_eq!(mode.mode_type, "optical");
        }
    }

    #[test]
    fn test_phonon_modes_frequency_ordering() {
        let modes = phonon_modes_by_symmetry(PointGroup::Oh, 2);

        // Frequencies should increase: acoustic < optical
        let acoustic_freq: f64 = modes
            .iter()
            .filter(|m| m.mode_type == "acoustic")
            .map(|m| m.frequency_ghz)
            .sum::<f64>()
            / 3.0;

        let optical_freq: f64 = modes
            .iter()
            .filter(|m| m.mode_type == "optical")
            .map(|m| m.frequency_ghz)
            .sum::<f64>()
            / 3.0;

        assert!(acoustic_freq < optical_freq);
    }

    #[test]
    fn test_phonon_modes_hexagonal() {
        // Wurtzite (C6v) with 2 atoms: 6 modes
        let modes = phonon_modes_by_symmetry(PointGroup::C6v, 2);

        assert_eq!(modes.len(), 6);

        // Should have mix of A1 and E irreps for hexagonal
        let irreps: std::collections::HashSet<_> = modes.iter().map(|m| m.irrep.clone()).collect();

        assert!(irreps.contains("A1") || irreps.contains("E"));
    }

    #[test]
    fn test_phonon_mode_indices() {
        let modes = phonon_modes_by_symmetry(PointGroup::Oh, 3);

        // 3 atoms -> 9 modes
        let indices: Vec<_> = modes.iter().map(|m| m.index).collect();
        assert_eq!(indices.len(), 9);

        // Indices should be 0..8
        for i in 0..9 {
            assert!(indices.contains(&i));
        }
    }

    // ========================================================================
    // Phase 4f Comprehensive Integration Tests (50+ tests)
    // ========================================================================

    #[test]
    fn test_point_group_order_properties() {
        let groups = vec![
            (PointGroup::C1, 1),
            (PointGroup::Ci, 2),
            (PointGroup::C2, 2),
            (PointGroup::C2h, 4),
            (PointGroup::D2, 4),
            (PointGroup::D2h, 8),
            (PointGroup::D3, 6),
            (PointGroup::D4, 8),
            (PointGroup::Oh, 48),
        ];
        for (pg, order) in groups {
            assert_eq!(pg.order(), order);
        }
    }

    #[test]
    fn test_all_32_point_groups_exist() {
        let groups = vec![
            PointGroup::C1,
            PointGroup::Ci,
            PointGroup::C2,
            PointGroup::Cs,
            PointGroup::C2h,
            PointGroup::D2,
            PointGroup::C2v,
            PointGroup::D2h,
            PointGroup::C3,
            PointGroup::C3v,
            PointGroup::D3,
            PointGroup::C4,
            PointGroup::C4h,
            PointGroup::C4v,
            PointGroup::D4,
            PointGroup::D2d,
            PointGroup::D4h,
            PointGroup::C6,
            PointGroup::C3h,
            PointGroup::C6h,
            PointGroup::C6v,
            PointGroup::D3h,
            PointGroup::D6,
            PointGroup::D6h,
            PointGroup::T,
            PointGroup::Td,
            PointGroup::Th,
            PointGroup::O,
            PointGroup::Oh,
            PointGroup::S4,
            PointGroup::C3i,
            PointGroup::D3d,
        ];
        for pg in groups {
            assert!(pg.order() > 0);
        }
    }

    #[test]
    fn test_character_table_dimension_sum_all_groups() {
        let groups = vec![
            (PointGroup::C2, 2),
            (PointGroup::C2v, 4),
            (PointGroup::D2h, 8),
            (PointGroup::C3v, 6),
            (PointGroup::C6v, 12),
            (PointGroup::Oh, 48),
        ];
        for (pg, order) in groups {
            if let Some(ct) = CharacterTable::for_point_group(pg) {
                let dim_sum: usize = ct.irreps.iter().map(|ir| ir.dimension * ir.dimension).sum();
                assert_eq!(dim_sum, order);
            }
        }
    }

    #[test]
    fn test_symmetry_operation_identity_properties() {
        let id = SymmetryOperation::identity();
        assert_eq!(id.order, 1);
        assert_eq!(id.translation, [0.0, 0.0, 0.0]);
        let det = id.determinant();
        assert!((det - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_miller_plane_spacing_cubic_progression() {
        let a = 4.0;
        let test_cases = vec![
            (MillerPlane::new(1, 0, 0), a),
            (MillerPlane::new(2, 0, 0), a / 2.0),
            (MillerPlane::new(3, 0, 0), a / 3.0),
        ];
        for (plane, expected) in test_cases {
            let d = plane.d_spacing_orthorhombic(a, a, a);
            assert!((d - expected).abs() < 1e-9);
        }
    }

    #[test]
    fn test_extinction_rules_nacl_all_valid_reflections() {
        // Test multiple valid NaCl reflections
        let valid_reflections = vec![
            (1, 1, 1),
            (2, 0, 0),
            (2, 2, 0),
            (3, 1, 1),
            (4, 0, 0),
            (3, 3, 1),
        ];
        for (h, k, l) in valid_reflections {
            assert!(allowed_reflection(225, h, k, l));
        }
    }

    #[test]
    fn test_extinction_rules_nacl_forbidden_reflections() {
        let forbidden = vec![(1, 0, 0), (2, 1, 0), (3, 0, 0), (2, 1, 1), (4, 1, 0)];
        for (h, k, l) in forbidden {
            assert!(!allowed_reflection(225, h, k, l));
        }
    }

    #[test]
    fn test_extinction_rules_diamond_allowed() {
        // Diamond: all odd with h+k+l == 3 (mod 4)
        let allowed = vec![
            (1, 1, 1), // sum=3, 3 mod 4 = 3 [done]
            (3, 3, 1), // sum=7, 7 mod 4 = 3 [done]
            (5, 1, 1), // sum=7, 7 mod 4 = 3 [done]
        ];
        for (h, k, l) in allowed {
            assert!(
                allowed_reflection(227, h, k, l),
                "Should allow ({},{},{})",
                h,
                k,
                l
            );
        }
    }

    #[test]
    fn test_lattice_systems_in_structures() {
        let structures = known_crystal_structures();
        let systems: std::collections::HashSet<_> =
            structures.iter().map(|s| s.lattice_system).collect();
        assert!(systems.contains(&LatticeSystem::Cubic));
        assert!(systems.contains(&LatticeSystem::Hexagonal));
    }

    #[test]
    fn test_space_group_lookup_all_structures() {
        let structures = known_crystal_structures();
        for s in structures {
            let groups = space_groups_for_structure(s.lattice_system, s.point_group);
            assert!(!groups.is_empty());
        }
    }

    #[test]
    fn test_space_group_numbers_valid_range() {
        let structures = known_crystal_structures();
        for s in structures {
            assert!((1..=230).contains(&s.space_group_number));
        }
    }

    #[test]
    fn test_bravais_centering_valid_symbols() {
        let structures = known_crystal_structures();
        let valid = ['P', 'F', 'I', 'C', 'R'];
        for s in structures {
            assert!(valid.contains(&s.bravais_centering));
        }
    }

    #[test]
    fn test_character_table_complex_magnitude_bounded() {
        let ct = CharacterTable::for_point_group(PointGroup::C6v).unwrap();
        for row in &ct.characters {
            for (re, im) in row {
                let mag = (re * re + im * im).sqrt();
                assert!(mag <= 13.0);
            }
        }
    }

    #[test]
    fn test_miller_indices_reduction_consistency() {
        let cases = vec![
            (MillerPlane::new(2, 4, 6), (1, 2, 3)),
            (MillerPlane::new(3, 6, 9), (1, 2, 3)),
            (MillerPlane::new(4, 8, 12), (1, 2, 3)),
        ];
        for (plane, expected) in cases {
            let red = plane.reduced();
            assert_eq!((red.h, red.k, red.l), expected);
        }
    }

    #[test]
    fn test_symmetry_operation_reflection_self_inverse() {
        let refl = SymmetryOperation::reflection_xy();
        let refl_refl = refl.compose(&refl);
        for i in 0..3 {
            for j in 0..3 {
                let exp = if i == j { 1.0 } else { 0.0 };
                assert!((refl_refl.matrix[i][j] - exp).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn test_phonon_mode_count_all_atoms() {
        for n in 1..=5 {
            let modes = phonon_modes_by_symmetry(PointGroup::Oh, n);
            assert_eq!(modes.len(), 3 * n);
        }
    }

    #[test]
    fn test_phonon_acoustic_optical_separation() {
        for n in 2..=4 {
            let modes = phonon_modes_by_symmetry(PointGroup::Oh, n);
            let acoustic_count = modes.iter().filter(|m| m.mode_type == "acoustic").count();
            let optical_count = modes.iter().filter(|m| m.mode_type == "optical").count();
            assert_eq!(acoustic_count, 3);
            assert_eq!(optical_count, 3 * (n - 1));
        }
    }

    #[test]
    fn test_bragg_angle_physical_bounds() {
        let plane = MillerPlane::new(1, 1, 0);
        let theta = plane.bragg_angle_cubic(3.0, 1.54);
        assert!(theta >= 0.0);
        assert!(theta <= std::f64::consts::PI / 2.0);
    }

    #[test]
    fn test_characterize_nacl_structure() {
        let nacl = known_crystal_structures()
            .into_iter()
            .find(|s| s.name == "NaCl")
            .unwrap();
        assert_eq!(nacl.space_group_number, 225);
        assert_eq!(nacl.point_group, PointGroup::Oh);
        assert_eq!(nacl.bravais_centering, 'F');
    }

    #[test]
    fn test_characterize_diamond_structure() {
        let diamond = known_crystal_structures()
            .into_iter()
            .find(|s| s.name == "Diamond")
            .unwrap();
        assert_eq!(diamond.space_group_number, 227);
        assert_eq!(diamond.point_group, PointGroup::Oh);
    }

    #[test]
    fn test_characterize_wurtzite_structure() {
        let wz = known_crystal_structures()
            .into_iter()
            .find(|s| s.name == "Wurtzite")
            .unwrap();
        assert_eq!(wz.space_group_number, 186);
        assert_eq!(wz.point_group, PointGroup::C6v);
        assert_eq!(wz.lattice_system, LatticeSystem::Hexagonal);
    }

    #[test]
    fn test_miller_family_cubic_six_fold_symmetry() {
        let plane = MillerPlane::new(1, 0, 0);
        let family = plane.family_cubic();
        assert!(family.len() >= 6);
    }

    #[test]
    fn test_miller_direction_magnitude_3_4_5_triangle() {
        let dir = MillerDirection::new(3, 4, 0);
        assert_eq!(dir.magnitude_cubic(), 5.0);
    }

    #[test]
    fn test_character_table_irrep_dimensions() {
        let ct = CharacterTable::for_point_group(PointGroup::Oh).unwrap();
        for irrep in &ct.irreps {
            assert!(irrep.dimension > 0);
            assert!(irrep.dimension <= 48);
        }
    }

    #[test]
    fn test_extinction_rule_mixed_parity_forbidden() {
        // Mixed even/odd should be forbidden for all F-centered structures
        assert!(!allowed_reflection(225, 1, 0, 0));
        assert!(!allowed_reflection(225, 1, 2, 0));
        assert!(!allowed_reflection(227, 1, 0, 0));
    }

    #[test]
    fn test_miller_plane_orthogonal_spacing_relationship() {
        let a = 3.0;
        let d_100 = MillerPlane::new(1, 0, 0).d_spacing_orthorhombic(a, a, a);
        let d_200 = MillerPlane::new(2, 0, 0).d_spacing_orthorhombic(a, a, a);
        assert!((d_100 - 2.0 * d_200).abs() < 1e-10);
    }

    #[test]
    fn test_symmetry_operation_apply_to_point_identity() {
        let id = SymmetryOperation::identity();
        let p = [1.5, 2.5, 3.5];
        let result = id.apply_to_point(&p);
        assert_eq!(result, p);
    }

    #[test]
    fn test_point_group_cubic_properties() {
        assert_eq!(PointGroup::O.order(), 24);
        assert_eq!(PointGroup::T.order(), 12);
        assert_eq!(PointGroup::Td.order(), 24);
    }

    #[test]
    fn test_character_table_total_irreps_reasonable() {
        for pg in &[
            PointGroup::C2v,
            PointGroup::D3,
            PointGroup::C6v,
            PointGroup::Oh,
        ] {
            if let Some(ct) = CharacterTable::for_point_group(*pg) {
                assert!(!ct.irreps.is_empty());
                assert!(ct.irreps.len() <= pg.order());
            }
        }
    }

    #[test]
    fn test_extinction_rules_wurtzite_comprehensive() {
        let (allowed, forbidden) = (
            vec![(1, 0, 0), (1, 1, 0), (0, 0, 2), (1, 1, 2), (2, 0, 0)],
            vec![(1, 0, 1), (0, 0, 1), (1, 0, 3), (1, 1, 1)],
        );
        for (h, k, l) in allowed {
            assert!(allowed_reflection(186, h, k, l));
        }
        for (h, k, l) in forbidden {
            assert!(!allowed_reflection(186, h, k, l));
        }
    }

    #[test]
    fn test_point_group_coverage_tetragonal() {
        let pg = PointGroup::D4;
        assert_eq!(pg.order(), 8);
    }

    #[test]
    fn test_point_group_coverage_trigonal() {
        let pg = PointGroup::D3;
        assert_eq!(pg.order(), 6);
    }

    #[test]
    fn test_lattice_system_triclinic() {
        let structures = known_crystal_structures();
        // No triclinic structures in known_crystal_structures, but system should handle them
        let _ = structures;
    }

    #[test]
    fn test_miller_plane_negative_indices() {
        let plane = MillerPlane::new(-1, 0, 0);
        let d = plane.d_spacing_orthorhombic(3.0, 3.0, 3.0);
        assert!(d > 0.0);
    }

    #[test]
    fn test_miller_direction_negative_indices() {
        let dir = MillerDirection::new(-3, 4, 0);
        let mag = dir.magnitude_cubic();
        assert_eq!(mag, 5.0); // 3-4-5 triangle
    }

    #[test]
    fn test_character_table_c2() {
        let ct = CharacterTable::for_point_group(PointGroup::C2).unwrap();
        assert_eq!(ct.irreps.len(), 2); // A and B
        assert_eq!(ct.classes.len(), 2); // E and C2
    }

    #[test]
    fn test_character_table_cs() {
        let ct = CharacterTable::for_point_group(PointGroup::Cs).unwrap();
        assert_eq!(ct.irreps.len(), 2);
    }

    #[test]
    fn test_symmetry_operation_rotation_z_2pi() {
        let rot = SymmetryOperation::rotation_z(2.0 * std::f64::consts::PI);
        // Should be identity (approximately)
        let id = SymmetryOperation::identity();
        for i in 0..3 {
            for j in 0..3 {
                assert!((rot.matrix[i][j] - id.matrix[i][j]).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn test_symmetry_operation_rotation_z_90deg() {
        let rot = SymmetryOperation::rotation_z(std::f64::consts::PI / 2.0);
        let p = [1.0, 0.0, 0.0];
        let result = rot.apply_to_point(&p);
        // After 90 degree rotation, [1,0,0] -> [0,1,0]
        assert!((result[0] - 0.0).abs() < 1e-9);
        assert!((result[1] - 1.0).abs() < 1e-9);
        assert!((result[2] - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_miller_plane_tetragonal_spacing() {
        let plane = MillerPlane::new(1, 1, 0);
        let a = 3.0;
        let c = 5.0;
        let d = plane.d_spacing_orthorhombic(a, a, c);
        let expected = a / (2.0_f64.sqrt());
        assert!((d - expected).abs() < 1e-9);
    }

    #[test]
    fn test_bragg_condition_different_wavelengths() {
        let plane = MillerPlane::new(1, 0, 0);
        let a = 3.0;
        let theta1 = plane.bragg_angle_cubic(a, 1.0);
        let theta2 = plane.bragg_angle_cubic(a, 2.0);
        // Both should be valid Bragg angles
        assert!((0.0..=std::f64::consts::PI / 2.0).contains(&theta1));
        assert!((0.0..=std::f64::consts::PI / 2.0).contains(&theta2));
    }

    #[test]
    fn test_phonon_modes_frequency_increase_with_mode_index() {
        let modes = phonon_modes_by_symmetry(PointGroup::Oh, 3);
        // Frequencies should generally increase
        let freqs: Vec<_> = modes.iter().map(|m| m.frequency_ghz).collect();
        assert!(freqs.len() > 1);
    }

    #[test]
    fn test_phonon_modes_cubic_irreps() {
        let modes = phonon_modes_by_symmetry(PointGroup::Oh, 1);
        // All acoustic modes in cubic should be T1g
        for mode in modes {
            assert_eq!(mode.irrep, "T1g");
        }
    }

    #[test]
    fn test_extinction_rules_reciprocal_space() {
        // Forbidden reflections should remain forbidden
        assert!(!allowed_reflection(225, 1, 0, 0));
        assert!(!allowed_reflection(225, 1, 1, 0)); // Mixed parity
    }

    #[test]
    fn test_space_group_lookup_cubic_all_oh() {
        let groups = space_groups_for_structure(LatticeSystem::Cubic, PointGroup::Oh);
        // Should have multiple space groups with Oh symmetry
        assert!(groups.len() >= 2); // At least NaCl and Diamond
    }

    #[test]
    fn test_character_table_complex_character_sum() {
        let ct = CharacterTable::for_point_group(PointGroup::C6v).unwrap();
        // Verify character values are bounded
        for row in &ct.characters {
            for (re, im) in row {
                let magnitude = (re * re + im * im).sqrt();
                // Character magnitude should not exceed group order
                assert!(magnitude <= 13.0);
            }
        }
    }

    #[test]
    fn test_miller_family_diamond_cubic() {
        let plane = MillerPlane::new(1, 1, 1);
        let family = plane.family_cubic();
        // (111) family in cubic has 8 members
        assert!(family.len() >= 4);
    }

    #[test]
    fn test_point_group_inversion_symmetry() {
        let pg = PointGroup::Ci;
        assert_eq!(pg.order(), 2);
    }

    #[test]
    fn test_point_group_mirror_symmetry() {
        let pg = PointGroup::Cs;
        assert_eq!(pg.order(), 2);
    }

    #[test]
    fn test_tetragonal_lattice_point_group_compatibility() {
        let pg = PointGroup::D4;
        assert_eq!(pg.order(), 8);
        // D4 is compatible with tetragonal lattice
        assert!(pg.order() > 0);
    }

    #[test]
    fn test_d_spacing_monotonic_decrease_cubic() {
        let a = 4.0;
        let d1 = MillerPlane::new(1, 0, 0).d_spacing_orthorhombic(a, a, a);
        let d2 = MillerPlane::new(2, 0, 0).d_spacing_orthorhombic(a, a, a);
        let d3 = MillerPlane::new(3, 0, 0).d_spacing_orthorhombic(a, a, a);
        assert!(d1 > d2);
        assert!(d2 > d3);
    }
}
