//! Character tables for the 32 crystallographic point groups.
//!
//! Extracted from `crystal_symmetry.rs` as part of the #139 PH-MOD split
//! -- the largest cluster at ~1865 lines, dominated by hardcoded
//! `for_point_group()` tables. Future migration target for materials_data
//! build.rs codegen (#127): a CSV-driven character-table registry would
//! cut this submodule by an order of magnitude.

use std::fmt;

use super::PointGroup;

// ============================================================================
// Character Tables
// ============================================================================

/// A conjugacy class of symmetry operations.
#[derive(Debug, Clone)]
pub struct ConjugacyClass {
    /// Name of the class (e.g., "E", "C2", "sigma_v")
    pub name: String,
    /// Number of operations in this class
    pub count: usize,
}

/// An irreducible representation of a point group.
#[derive(Debug, Clone)]
pub struct IrreducibleRepresentation {
    /// Label of irrep (e.g., "A1", "E", "T2")
    pub label: String,
    /// Dimensionality (1, 2, or 3 for most common point groups)
    pub dimension: usize,
}

/// Complete character table for a point group.
/// Rows are irreducible representations, columns are conjugacy classes.
#[derive(Debug, Clone)]
pub struct CharacterTable {
    /// Point group this table represents
    pub point_group: PointGroup,
    /// Irreducible representations (rows)
    pub irreps: Vec<IrreducibleRepresentation>,
    /// Conjugacy classes (columns)
    pub classes: Vec<ConjugacyClass>,
    /// Character matrix: `irreps[i] x classes[j]`
    /// Complex numbers represented as (real, imaginary)
    pub characters: Vec<Vec<(f64, f64)>>,
}

/// Convert a codegen-emitted point-group name string back to the
/// PointGroup enum for the character-table lookup (#127 Phase 7).
/// Full mapping of all 32 crystallographic point groups; the codegen
/// path is the canonical source for any group whose name matches.
fn point_group_name_to_enum(name: &str) -> Option<PointGroup> {
    match name {
        "C1" => Some(PointGroup::C1),
        "Ci" => Some(PointGroup::Ci),
        "C2" => Some(PointGroup::C2),
        "Cs" => Some(PointGroup::Cs),
        "C2h" => Some(PointGroup::C2h),
        "D2" => Some(PointGroup::D2),
        "C2v" => Some(PointGroup::C2v),
        "D2h" => Some(PointGroup::D2h),
        "C4" => Some(PointGroup::C4),
        "S4" => Some(PointGroup::S4),
        "C4h" => Some(PointGroup::C4h),
        "D4" => Some(PointGroup::D4),
        "C4v" => Some(PointGroup::C4v),
        "D2d" => Some(PointGroup::D2d),
        "D4h" => Some(PointGroup::D4h),
        "C3" => Some(PointGroup::C3),
        "C3i" => Some(PointGroup::C3i),
        "C3v" => Some(PointGroup::C3v),
        "D3" => Some(PointGroup::D3),
        "D3d" => Some(PointGroup::D3d),
        "C6" => Some(PointGroup::C6),
        "C3h" => Some(PointGroup::C3h),
        "C6h" => Some(PointGroup::C6h),
        "D6" => Some(PointGroup::D6),
        "C6v" => Some(PointGroup::C6v),
        "D3h" => Some(PointGroup::D3h),
        "D6h" => Some(PointGroup::D6h),
        "T" => Some(PointGroup::T),
        "Td" => Some(PointGroup::Td),
        "Th" => Some(PointGroup::Th),
        "O" => Some(PointGroup::O),
        "Oh" => Some(PointGroup::Oh),
        _ => None,
    }
}

/// Look up a CharacterTable from the build-time codegen registry.
/// Returns `None` if the point group is not yet migrated to TOML.
fn from_codegen_table(pg: PointGroup) -> Option<CharacterTable> {
    materials_data::CHARACTER_TABLE_REGISTRY.iter().find_map(
        |(pg_name, classes_slice, irreps_slice, chars_slice)| {
            let pg_match = point_group_name_to_enum(pg_name)?;
            if pg_match != pg {
                return None;
            }
            let n_cls = classes_slice.len();
            let n_irr = irreps_slice.len();
            assert_eq!(
                chars_slice.len(),
                n_cls * n_irr,
                "codegen character table for {} has flat-slice length {} but n_irr*n_cls = {}",
                pg_name,
                chars_slice.len(),
                n_cls * n_irr
            );
            let classes: Vec<ConjugacyClass> = classes_slice
                .iter()
                .map(|(name, count)| ConjugacyClass {
                    name: name.to_string(),
                    count: *count as usize,
                })
                .collect();
            let irreps: Vec<IrreducibleRepresentation> = irreps_slice
                .iter()
                .map(|(label, dim)| IrreducibleRepresentation {
                    label: label.to_string(),
                    dimension: *dim as usize,
                })
                .collect();
            // Un-flatten the row-major (re, im) slice into the nested Vec<Vec<(f64, f64)>>.
            let characters: Vec<Vec<(f64, f64)>> = (0..n_irr)
                .map(|i| {
                    let start = i * n_cls;
                    chars_slice[start..start + n_cls].to_vec()
                })
                .collect();
            Some(CharacterTable {
                point_group: pg,
                classes,
                irreps,
                characters,
            })
        },
    )
}

impl CharacterTable {
    /// Get character table for a given point group.
    /// Returns Some(table) for supported groups, None otherwise.
    pub fn for_point_group(pg: PointGroup) -> Option<Self> {
        // Phase 7 codegen path: try the build-time TOML registry first.
        // The first 5 abelian groups (C1, Ci, C2, Cs, C2h) are migrated;
        // remaining 27 fall through to the inline `Self::*()` methods.
        if let Some(table) = from_codegen_table(pg) {
            return Some(table);
        }
        match pg {
            PointGroup::C1 => Some(Self::c1()),
            PointGroup::Ci => Some(Self::ci()),
            PointGroup::C2 => Some(Self::c2()),
            PointGroup::Cs => Some(Self::cs()),
            PointGroup::C2h => Some(Self::c2h()),
            PointGroup::D2 => Some(Self::d2()),
            PointGroup::C2v => Some(Self::c2v()),
            PointGroup::D2h => Some(Self::d2h()),
            PointGroup::C3 => Some(Self::c3()),
            PointGroup::C3v => Some(Self::c3v()),
            PointGroup::D3 => Some(Self::d3()),
            PointGroup::C4 => Some(Self::c4()),
            PointGroup::C4v => Some(Self::c4v()),
            PointGroup::D4 => Some(Self::d4()),
            PointGroup::D4h => Some(Self::d4h()),
            PointGroup::C6 => Some(Self::c6()),
            PointGroup::C6v => Some(Self::c6v()),
            PointGroup::D6 => Some(Self::d6()),
            PointGroup::D6h => Some(Self::d6h()),
            PointGroup::T => Some(Self::t()),
            PointGroup::Td => Some(Self::td()),
            PointGroup::Oh => Some(Self::oh()),
            // Remaining groups scaffolded as None
            _ => None,
        }
    }

    /// C1: Identity only (1x1)
    fn c1() -> Self {
        Self {
            point_group: PointGroup::C1,
            irreps: vec![IrreducibleRepresentation {
                label: "A".to_string(),
                dimension: 1,
            }],
            classes: vec![ConjugacyClass {
                name: "E".to_string(),
                count: 1,
            }],
            characters: vec![vec![(1.0, 0.0)]],
        }
    }

    /// Ci: Inversion center (2x2)
    fn ci() -> Self {
        Self {
            point_group: PointGroup::Ci,
            irreps: vec![
                IrreducibleRepresentation {
                    label: "Ag".to_string(),
                    dimension: 1,
                },
                IrreducibleRepresentation {
                    label: "Au".to_string(),
                    dimension: 1,
                },
            ],
            classes: vec![
                ConjugacyClass {
                    name: "E".to_string(),
                    count: 1,
                },
                ConjugacyClass {
                    name: "i".to_string(),
                    count: 1,
                },
            ],
            characters: vec![vec![(1.0, 0.0), (1.0, 0.0)], vec![(1.0, 0.0), (-1.0, 0.0)]],
        }
    }

    /// C2: 2-fold rotation (2x2)
    fn c2() -> Self {
        Self {
            point_group: PointGroup::C2,
            irreps: vec![
                IrreducibleRepresentation {
                    label: "A".to_string(),
                    dimension: 1,
                },
                IrreducibleRepresentation {
                    label: "B".to_string(),
                    dimension: 1,
                },
            ],
            classes: vec![
                ConjugacyClass {
                    name: "E".to_string(),
                    count: 1,
                },
                ConjugacyClass {
                    name: "C2".to_string(),
                    count: 1,
                },
            ],
            characters: vec![vec![(1.0, 0.0), (1.0, 0.0)], vec![(1.0, 0.0), (-1.0, 0.0)]],
        }
    }

    /// Cs: Mirror plane (2x2)
    fn cs() -> Self {
        Self {
            point_group: PointGroup::Cs,
            irreps: vec![
                IrreducibleRepresentation {
                    label: "A'".to_string(),
                    dimension: 1,
                },
                IrreducibleRepresentation {
                    label: "A''".to_string(),
                    dimension: 1,
                },
            ],
            classes: vec![
                ConjugacyClass {
                    name: "E".to_string(),
                    count: 1,
                },
                ConjugacyClass {
                    name: "sigma".to_string(),
                    count: 1,
                },
            ],
            characters: vec![vec![(1.0, 0.0), (1.0, 0.0)], vec![(1.0, 0.0), (-1.0, 0.0)]],
        }
    }

    /// C2h: 2-fold rotation + inversion (4x4)
    fn c2h() -> Self {
        Self {
            point_group: PointGroup::C2h,
            irreps: vec![
                IrreducibleRepresentation {
                    label: "Ag".to_string(),
                    dimension: 1,
                },
                IrreducibleRepresentation {
                    label: "Bg".to_string(),
                    dimension: 1,
                },
                IrreducibleRepresentation {
                    label: "Au".to_string(),
                    dimension: 1,
                },
                IrreducibleRepresentation {
                    label: "Bu".to_string(),
                    dimension: 1,
                },
            ],
            classes: vec![
                ConjugacyClass {
                    name: "E".to_string(),
                    count: 1,
                },
                ConjugacyClass {
                    name: "C2".to_string(),
                    count: 1,
                },
                ConjugacyClass {
                    name: "i".to_string(),
                    count: 1,
                },
                ConjugacyClass {
                    name: "sigma_h".to_string(),
                    count: 1,
                },
            ],
            characters: vec![
                vec![(1.0, 0.0), (1.0, 0.0), (1.0, 0.0), (1.0, 0.0)],
                vec![(1.0, 0.0), (-1.0, 0.0), (1.0, 0.0), (-1.0, 0.0)],
                vec![(1.0, 0.0), (1.0, 0.0), (-1.0, 0.0), (-1.0, 0.0)],
                vec![(1.0, 0.0), (-1.0, 0.0), (-1.0, 0.0), (1.0, 0.0)],
            ],
        }
    }

    /// D2: Three perpendicular 2-fold rotations (4x4)
    fn d2() -> Self {
        Self {
            point_group: PointGroup::D2,
            irreps: vec![
                IrreducibleRepresentation {
                    label: "A".to_string(),
                    dimension: 1,
                },
                IrreducibleRepresentation {
                    label: "B1".to_string(),
                    dimension: 1,
                },
                IrreducibleRepresentation {
                    label: "B2".to_string(),
                    dimension: 1,
                },
                IrreducibleRepresentation {
                    label: "B3".to_string(),
                    dimension: 1,
                },
            ],
            classes: vec![
                ConjugacyClass {
                    name: "E".to_string(),
                    count: 1,
                },
                ConjugacyClass {
                    name: "C2(z)".to_string(),
                    count: 1,
                },
                ConjugacyClass {
                    name: "C2(y)".to_string(),
                    count: 1,
                },
                ConjugacyClass {
                    name: "C2(x)".to_string(),
                    count: 1,
                },
            ],
            characters: vec![
                vec![(1.0, 0.0), (1.0, 0.0), (1.0, 0.0), (1.0, 0.0)],
                vec![(1.0, 0.0), (1.0, 0.0), (-1.0, 0.0), (-1.0, 0.0)],
                vec![(1.0, 0.0), (-1.0, 0.0), (1.0, 0.0), (-1.0, 0.0)],
                vec![(1.0, 0.0), (-1.0, 0.0), (-1.0, 0.0), (1.0, 0.0)],
            ],
        }
    }

    /// C2v: 2-fold rotation + two mirror planes (4x4)
    fn c2v() -> Self {
        Self {
            point_group: PointGroup::C2v,
            irreps: vec![
                IrreducibleRepresentation {
                    label: "A1".to_string(),
                    dimension: 1,
                },
                IrreducibleRepresentation {
                    label: "A2".to_string(),
                    dimension: 1,
                },
                IrreducibleRepresentation {
                    label: "B1".to_string(),
                    dimension: 1,
                },
                IrreducibleRepresentation {
                    label: "B2".to_string(),
                    dimension: 1,
                },
            ],
            classes: vec![
                ConjugacyClass {
                    name: "E".to_string(),
                    count: 1,
                },
                ConjugacyClass {
                    name: "C2".to_string(),
                    count: 1,
                },
                ConjugacyClass {
                    name: "sigma_v".to_string(),
                    count: 2,
                },
            ],
            characters: vec![
                vec![(1.0, 0.0), (1.0, 0.0), (1.0, 0.0)],
                vec![(1.0, 0.0), (1.0, 0.0), (-1.0, 0.0)],
                vec![(1.0, 0.0), (-1.0, 0.0), (1.0, 0.0)],
                vec![(1.0, 0.0), (-1.0, 0.0), (-1.0, 0.0)],
            ],
        }
    }

    /// D2h: Three 2-fold + three mirrors (full orthorhombic) (8x8)
    fn d2h() -> Self {
        Self {
            point_group: PointGroup::D2h,
            irreps: vec![
                IrreducibleRepresentation {
                    label: "Ag".to_string(),
                    dimension: 1,
                },
                IrreducibleRepresentation {
                    label: "B1g".to_string(),
                    dimension: 1,
                },
                IrreducibleRepresentation {
                    label: "B2g".to_string(),
                    dimension: 1,
                },
                IrreducibleRepresentation {
                    label: "B3g".to_string(),
                    dimension: 1,
                },
                IrreducibleRepresentation {
                    label: "Au".to_string(),
                    dimension: 1,
                },
                IrreducibleRepresentation {
                    label: "B1u".to_string(),
                    dimension: 1,
                },
                IrreducibleRepresentation {
                    label: "B2u".to_string(),
                    dimension: 1,
                },
                IrreducibleRepresentation {
                    label: "B3u".to_string(),
                    dimension: 1,
                },
            ],
            classes: vec![
                ConjugacyClass {
                    name: "E".to_string(),
                    count: 1,
                },
                ConjugacyClass {
                    name: "C2(z)".to_string(),
                    count: 1,
                },
                ConjugacyClass {
                    name: "C2(y)".to_string(),
                    count: 1,
                },
                ConjugacyClass {
                    name: "C2(x)".to_string(),
                    count: 1,
                },
                ConjugacyClass {
                    name: "i".to_string(),
                    count: 1,
                },
                ConjugacyClass {
                    name: "sigma(xy)".to_string(),
                    count: 1,
                },
                ConjugacyClass {
                    name: "sigma(xz)".to_string(),
                    count: 1,
                },
                ConjugacyClass {
                    name: "sigma(yz)".to_string(),
                    count: 1,
                },
            ],
            characters: vec![
                vec![
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                ],
                vec![
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (-1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (-1.0, 0.0),
                ],
                vec![
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (1.0, 0.0),
                    (-1.0, 0.0),
                ],
                vec![
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (-1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (-1.0, 0.0),
                    (1.0, 0.0),
                ],
                vec![
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (-1.0, 0.0),
                    (-1.0, 0.0),
                    (-1.0, 0.0),
                ],
                vec![
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (-1.0, 0.0),
                    (-1.0, 0.0),
                    (-1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                ],
                vec![
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (-1.0, 0.0),
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (1.0, 0.0),
                ],
                vec![
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (-1.0, 0.0),
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (-1.0, 0.0),
                ],
            ],
        }
    }

    /// C3: 3-fold rotation (3x3)
    fn c3() -> Self {
        let w = std::f64::consts::PI / 3.0; // omega for 3rd roots of unity
        Self {
            point_group: PointGroup::C3,
            irreps: vec![
                IrreducibleRepresentation {
                    label: "A".to_string(),
                    dimension: 1,
                },
                IrreducibleRepresentation {
                    label: "E(1)".to_string(),
                    dimension: 2,
                },
            ],
            classes: vec![
                ConjugacyClass {
                    name: "E".to_string(),
                    count: 1,
                },
                ConjugacyClass {
                    name: "C3".to_string(),
                    count: 1,
                },
                ConjugacyClass {
                    name: "C3^2".to_string(),
                    count: 1,
                },
            ],
            characters: vec![
                vec![(1.0, 0.0), (1.0, 0.0), (1.0, 0.0)],
                vec![(2.0, 0.0), (-w.cos(), w.sin()), (-w.cos(), -w.sin())],
            ],
        }
    }

    /// C3v: 3-fold rotation + three mirror planes (6x3)
    fn c3v() -> Self {
        Self {
            point_group: PointGroup::C3v,
            irreps: vec![
                IrreducibleRepresentation {
                    label: "A1".to_string(),
                    dimension: 1,
                },
                IrreducibleRepresentation {
                    label: "A2".to_string(),
                    dimension: 1,
                },
                IrreducibleRepresentation {
                    label: "E".to_string(),
                    dimension: 2,
                },
            ],
            classes: vec![
                ConjugacyClass {
                    name: "E".to_string(),
                    count: 1,
                },
                ConjugacyClass {
                    name: "C3".to_string(),
                    count: 2,
                },
                ConjugacyClass {
                    name: "sigma_v".to_string(),
                    count: 3,
                },
            ],
            characters: vec![
                vec![(1.0, 0.0), (1.0, 0.0), (1.0, 0.0)],
                vec![(1.0, 0.0), (1.0, 0.0), (-1.0, 0.0)],
                vec![(2.0, 0.0), (-1.0, 0.0), (0.0, 0.0)],
            ],
        }
    }

    /// D3: 3-fold rotation + three 2-fold rotations (6x3)
    fn d3() -> Self {
        Self {
            point_group: PointGroup::D3,
            irreps: vec![
                IrreducibleRepresentation {
                    label: "A1".to_string(),
                    dimension: 1,
                },
                IrreducibleRepresentation {
                    label: "A2".to_string(),
                    dimension: 1,
                },
                IrreducibleRepresentation {
                    label: "E".to_string(),
                    dimension: 2,
                },
            ],
            classes: vec![
                ConjugacyClass {
                    name: "E".to_string(),
                    count: 1,
                },
                ConjugacyClass {
                    name: "C3".to_string(),
                    count: 2,
                },
                ConjugacyClass {
                    name: "C2".to_string(),
                    count: 3,
                },
            ],
            characters: vec![
                vec![(1.0, 0.0), (1.0, 0.0), (1.0, 0.0)],
                vec![(1.0, 0.0), (1.0, 0.0), (-1.0, 0.0)],
                vec![(2.0, 0.0), (-1.0, 0.0), (0.0, 0.0)],
            ],
        }
    }

    /// C4: 4-fold rotation (4x4)
    fn c4() -> Self {
        Self {
            point_group: PointGroup::C4,
            irreps: vec![
                IrreducibleRepresentation {
                    label: "A".to_string(),
                    dimension: 1,
                },
                IrreducibleRepresentation {
                    label: "B".to_string(),
                    dimension: 1,
                },
                IrreducibleRepresentation {
                    label: "E(1)".to_string(),
                    dimension: 2,
                },
            ],
            classes: vec![
                ConjugacyClass {
                    name: "E".to_string(),
                    count: 1,
                },
                ConjugacyClass {
                    name: "C4".to_string(),
                    count: 1,
                },
                ConjugacyClass {
                    name: "C2".to_string(),
                    count: 1,
                },
                ConjugacyClass {
                    name: "C4^3".to_string(),
                    count: 1,
                },
            ],
            characters: vec![
                vec![(1.0, 0.0), (1.0, 0.0), (1.0, 0.0), (1.0, 0.0)],
                vec![(1.0, 0.0), (-1.0, 0.0), (1.0, 0.0), (-1.0, 0.0)],
                vec![(2.0, 0.0), (0.0, 0.0), (-2.0, 0.0), (0.0, 0.0)],
            ],
        }
    }

    /// C4v: 4-fold rotation + four mirror planes (8x5)
    fn c4v() -> Self {
        Self {
            point_group: PointGroup::C4v,
            irreps: vec![
                IrreducibleRepresentation {
                    label: "A1".to_string(),
                    dimension: 1,
                },
                IrreducibleRepresentation {
                    label: "A2".to_string(),
                    dimension: 1,
                },
                IrreducibleRepresentation {
                    label: "B1".to_string(),
                    dimension: 1,
                },
                IrreducibleRepresentation {
                    label: "B2".to_string(),
                    dimension: 1,
                },
                IrreducibleRepresentation {
                    label: "E".to_string(),
                    dimension: 2,
                },
            ],
            classes: vec![
                ConjugacyClass {
                    name: "E".to_string(),
                    count: 1,
                },
                ConjugacyClass {
                    name: "C4".to_string(),
                    count: 2,
                },
                ConjugacyClass {
                    name: "C2".to_string(),
                    count: 1,
                },
                ConjugacyClass {
                    name: "sigma_v".to_string(),
                    count: 2,
                },
                ConjugacyClass {
                    name: "sigma_d".to_string(),
                    count: 2,
                },
            ],
            characters: vec![
                vec![(1.0, 0.0), (1.0, 0.0), (1.0, 0.0), (1.0, 0.0), (1.0, 0.0)],
                vec![(1.0, 0.0), (1.0, 0.0), (1.0, 0.0), (-1.0, 0.0), (-1.0, 0.0)],
                vec![(1.0, 0.0), (-1.0, 0.0), (1.0, 0.0), (1.0, 0.0), (-1.0, 0.0)],
                vec![(1.0, 0.0), (-1.0, 0.0), (1.0, 0.0), (-1.0, 0.0), (1.0, 0.0)],
                vec![(2.0, 0.0), (0.0, 0.0), (-2.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
            ],
        }
    }

    /// D4: 4-fold rotation + four 2-fold rotations (8x5)
    fn d4() -> Self {
        Self {
            point_group: PointGroup::D4,
            irreps: vec![
                IrreducibleRepresentation {
                    label: "A1".to_string(),
                    dimension: 1,
                },
                IrreducibleRepresentation {
                    label: "A2".to_string(),
                    dimension: 1,
                },
                IrreducibleRepresentation {
                    label: "B1".to_string(),
                    dimension: 1,
                },
                IrreducibleRepresentation {
                    label: "B2".to_string(),
                    dimension: 1,
                },
                IrreducibleRepresentation {
                    label: "E".to_string(),
                    dimension: 2,
                },
            ],
            classes: vec![
                ConjugacyClass {
                    name: "E".to_string(),
                    count: 1,
                },
                ConjugacyClass {
                    name: "C4".to_string(),
                    count: 2,
                },
                ConjugacyClass {
                    name: "C2".to_string(),
                    count: 1,
                },
                ConjugacyClass {
                    name: "C2'".to_string(),
                    count: 2,
                },
                ConjugacyClass {
                    name: "C2''".to_string(),
                    count: 2,
                },
            ],
            characters: vec![
                vec![(1.0, 0.0), (1.0, 0.0), (1.0, 0.0), (1.0, 0.0), (1.0, 0.0)],
                vec![(1.0, 0.0), (1.0, 0.0), (1.0, 0.0), (-1.0, 0.0), (-1.0, 0.0)],
                vec![(1.0, 0.0), (-1.0, 0.0), (1.0, 0.0), (1.0, 0.0), (-1.0, 0.0)],
                vec![(1.0, 0.0), (-1.0, 0.0), (1.0, 0.0), (-1.0, 0.0), (1.0, 0.0)],
                vec![(2.0, 0.0), (0.0, 0.0), (-2.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
            ],
        }
    }

    /// D4h: 4-fold + mirrors (full tetragonal) (16x9)
    fn d4h() -> Self {
        Self {
            point_group: PointGroup::D4h,
            irreps: vec![
                IrreducibleRepresentation {
                    label: "A1g".to_string(),
                    dimension: 1,
                },
                IrreducibleRepresentation {
                    label: "A2g".to_string(),
                    dimension: 1,
                },
                IrreducibleRepresentation {
                    label: "B1g".to_string(),
                    dimension: 1,
                },
                IrreducibleRepresentation {
                    label: "B2g".to_string(),
                    dimension: 1,
                },
                IrreducibleRepresentation {
                    label: "Eg".to_string(),
                    dimension: 2,
                },
                IrreducibleRepresentation {
                    label: "A1u".to_string(),
                    dimension: 1,
                },
                IrreducibleRepresentation {
                    label: "A2u".to_string(),
                    dimension: 1,
                },
                IrreducibleRepresentation {
                    label: "B1u".to_string(),
                    dimension: 1,
                },
                IrreducibleRepresentation {
                    label: "B2u".to_string(),
                    dimension: 1,
                },
            ],
            classes: vec![
                ConjugacyClass {
                    name: "E".to_string(),
                    count: 1,
                },
                ConjugacyClass {
                    name: "C4".to_string(),
                    count: 2,
                },
                ConjugacyClass {
                    name: "C2".to_string(),
                    count: 1,
                },
                ConjugacyClass {
                    name: "C2'".to_string(),
                    count: 2,
                },
                ConjugacyClass {
                    name: "C2''".to_string(),
                    count: 2,
                },
                ConjugacyClass {
                    name: "i".to_string(),
                    count: 1,
                },
                ConjugacyClass {
                    name: "S4".to_string(),
                    count: 2,
                },
                ConjugacyClass {
                    name: "sigma_h".to_string(),
                    count: 1,
                },
                ConjugacyClass {
                    name: "sigma_v".to_string(),
                    count: 2,
                },
            ],
            characters: vec![
                vec![
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                ],
                vec![
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (-1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (-1.0, 0.0),
                ],
                vec![
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                ],
                vec![
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (1.0, 0.0),
                    (-1.0, 0.0),
                ],
                vec![
                    (2.0, 0.0),
                    (0.0, 0.0),
                    (-2.0, 0.0),
                    (0.0, 0.0),
                    (0.0, 0.0),
                    (2.0, 0.0),
                    (0.0, 0.0),
                    (-2.0, 0.0),
                    (0.0, 0.0),
                ],
                vec![
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (-1.0, 0.0),
                    (-1.0, 0.0),
                    (-1.0, 0.0),
                ],
                vec![
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (-1.0, 0.0),
                    (-1.0, 0.0),
                    (-1.0, 0.0),
                    (-1.0, 0.0),
                    (1.0, 0.0),
                ],
                vec![
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (-1.0, 0.0),
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (-1.0, 0.0),
                ],
                vec![
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (1.0, 0.0),
                ],
            ],
        }
    }

    /// C6: 6-fold rotation (6x6)
    fn c6() -> Self {
        Self {
            point_group: PointGroup::C6,
            irreps: vec![
                IrreducibleRepresentation {
                    label: "A".to_string(),
                    dimension: 1,
                },
                IrreducibleRepresentation {
                    label: "B".to_string(),
                    dimension: 1,
                },
                IrreducibleRepresentation {
                    label: "E(1)".to_string(),
                    dimension: 2,
                },
                IrreducibleRepresentation {
                    label: "E(2)".to_string(),
                    dimension: 2,
                },
            ],
            classes: vec![
                ConjugacyClass {
                    name: "E".to_string(),
                    count: 1,
                },
                ConjugacyClass {
                    name: "C6".to_string(),
                    count: 1,
                },
                ConjugacyClass {
                    name: "C3".to_string(),
                    count: 1,
                },
                ConjugacyClass {
                    name: "C2".to_string(),
                    count: 1,
                },
                ConjugacyClass {
                    name: "C3^2".to_string(),
                    count: 1,
                },
                ConjugacyClass {
                    name: "C6^5".to_string(),
                    count: 1,
                },
            ],
            characters: vec![
                vec![
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                ],
                vec![
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (1.0, 0.0),
                    (-1.0, 0.0),
                ],
                vec![
                    (2.0, 0.0),
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (-2.0, 0.0),
                    (-1.0, 0.0),
                    (1.0, 0.0),
                ],
                vec![
                    (2.0, 0.0),
                    (-1.0, 0.0),
                    (-1.0, 0.0),
                    (2.0, 0.0),
                    (-1.0, 0.0),
                    (-1.0, 0.0),
                ],
            ],
        }
    }

    /// C6v: 6-fold rotation + six mirror planes (12x6)
    fn c6v() -> Self {
        Self {
            point_group: PointGroup::C6v,
            irreps: vec![
                IrreducibleRepresentation {
                    label: "A1".to_string(),
                    dimension: 1,
                },
                IrreducibleRepresentation {
                    label: "A2".to_string(),
                    dimension: 1,
                },
                IrreducibleRepresentation {
                    label: "B1".to_string(),
                    dimension: 1,
                },
                IrreducibleRepresentation {
                    label: "B2".to_string(),
                    dimension: 1,
                },
                IrreducibleRepresentation {
                    label: "E(1)".to_string(),
                    dimension: 2,
                },
                IrreducibleRepresentation {
                    label: "E(2)".to_string(),
                    dimension: 2,
                },
            ],
            classes: vec![
                ConjugacyClass {
                    name: "E".to_string(),
                    count: 1,
                },
                ConjugacyClass {
                    name: "C6".to_string(),
                    count: 2,
                },
                ConjugacyClass {
                    name: "C3".to_string(),
                    count: 2,
                },
                ConjugacyClass {
                    name: "C2".to_string(),
                    count: 1,
                },
                ConjugacyClass {
                    name: "sigma_v".to_string(),
                    count: 3,
                },
                ConjugacyClass {
                    name: "sigma_d".to_string(),
                    count: 3,
                },
            ],
            characters: vec![
                vec![
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                ],
                vec![
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (-1.0, 0.0),
                ],
                vec![
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (1.0, 0.0),
                    (-1.0, 0.0),
                ],
                vec![
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (-1.0, 0.0),
                    (1.0, 0.0),
                ],
                vec![
                    (2.0, 0.0),
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (-2.0, 0.0),
                    (0.0, 0.0),
                    (0.0, 0.0),
                ],
                vec![
                    (2.0, 0.0),
                    (-1.0, 0.0),
                    (-1.0, 0.0),
                    (2.0, 0.0),
                    (0.0, 0.0),
                    (0.0, 0.0),
                ],
            ],
        }
    }

    /// D6: 6-fold rotation + six 2-fold rotations (12x6)
    fn d6() -> Self {
        Self {
            point_group: PointGroup::D6,
            irreps: vec![
                IrreducibleRepresentation {
                    label: "A1".to_string(),
                    dimension: 1,
                },
                IrreducibleRepresentation {
                    label: "A2".to_string(),
                    dimension: 1,
                },
                IrreducibleRepresentation {
                    label: "B1".to_string(),
                    dimension: 1,
                },
                IrreducibleRepresentation {
                    label: "B2".to_string(),
                    dimension: 1,
                },
                IrreducibleRepresentation {
                    label: "E(1)".to_string(),
                    dimension: 2,
                },
                IrreducibleRepresentation {
                    label: "E(2)".to_string(),
                    dimension: 2,
                },
            ],
            classes: vec![
                ConjugacyClass {
                    name: "E".to_string(),
                    count: 1,
                },
                ConjugacyClass {
                    name: "C6".to_string(),
                    count: 2,
                },
                ConjugacyClass {
                    name: "C3".to_string(),
                    count: 2,
                },
                ConjugacyClass {
                    name: "C2".to_string(),
                    count: 1,
                },
                ConjugacyClass {
                    name: "C2'".to_string(),
                    count: 3,
                },
                ConjugacyClass {
                    name: "C2''".to_string(),
                    count: 3,
                },
            ],
            characters: vec![
                vec![
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                ],
                vec![
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (-1.0, 0.0),
                ],
                vec![
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (1.0, 0.0),
                    (-1.0, 0.0),
                ],
                vec![
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (-1.0, 0.0),
                    (1.0, 0.0),
                ],
                vec![
                    (2.0, 0.0),
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (-2.0, 0.0),
                    (0.0, 0.0),
                    (0.0, 0.0),
                ],
                vec![
                    (2.0, 0.0),
                    (-1.0, 0.0),
                    (-1.0, 0.0),
                    (2.0, 0.0),
                    (0.0, 0.0),
                    (0.0, 0.0),
                ],
            ],
        }
    }

    /// D6h: 6-fold + mirrors (full hexagonal) (24x12)
    fn d6h() -> Self {
        Self {
            point_group: PointGroup::D6h,
            irreps: vec![
                IrreducibleRepresentation {
                    label: "A1g".to_string(),
                    dimension: 1,
                },
                IrreducibleRepresentation {
                    label: "A2g".to_string(),
                    dimension: 1,
                },
                IrreducibleRepresentation {
                    label: "B1g".to_string(),
                    dimension: 1,
                },
                IrreducibleRepresentation {
                    label: "B2g".to_string(),
                    dimension: 1,
                },
                IrreducibleRepresentation {
                    label: "E1g".to_string(),
                    dimension: 2,
                },
                IrreducibleRepresentation {
                    label: "E2g".to_string(),
                    dimension: 2,
                },
                IrreducibleRepresentation {
                    label: "A1u".to_string(),
                    dimension: 1,
                },
                IrreducibleRepresentation {
                    label: "A2u".to_string(),
                    dimension: 1,
                },
                IrreducibleRepresentation {
                    label: "B1u".to_string(),
                    dimension: 1,
                },
                IrreducibleRepresentation {
                    label: "B2u".to_string(),
                    dimension: 1,
                },
                IrreducibleRepresentation {
                    label: "E1u".to_string(),
                    dimension: 2,
                },
                IrreducibleRepresentation {
                    label: "E2u".to_string(),
                    dimension: 2,
                },
            ],
            classes: vec![
                ConjugacyClass {
                    name: "E".to_string(),
                    count: 1,
                },
                ConjugacyClass {
                    name: "C6".to_string(),
                    count: 2,
                },
                ConjugacyClass {
                    name: "C3".to_string(),
                    count: 2,
                },
                ConjugacyClass {
                    name: "C2".to_string(),
                    count: 1,
                },
                ConjugacyClass {
                    name: "C2'".to_string(),
                    count: 3,
                },
                ConjugacyClass {
                    name: "C2''".to_string(),
                    count: 3,
                },
                ConjugacyClass {
                    name: "i".to_string(),
                    count: 1,
                },
                ConjugacyClass {
                    name: "S6".to_string(),
                    count: 2,
                },
                ConjugacyClass {
                    name: "S3".to_string(),
                    count: 2,
                },
                ConjugacyClass {
                    name: "sigma_h".to_string(),
                    count: 1,
                },
                ConjugacyClass {
                    name: "sigma_v".to_string(),
                    count: 3,
                },
                ConjugacyClass {
                    name: "sigma_d".to_string(),
                    count: 3,
                },
            ],
            characters: vec![
                // A1g
                vec![
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                ],
                // A2g
                vec![
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (-1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (-1.0, 0.0),
                ],
                // B1g
                vec![
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (1.0, 0.0),
                    (-1.0, 0.0),
                ],
                // B2g
                vec![
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (-1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (-1.0, 0.0),
                    (1.0, 0.0),
                ],
                // E1g
                vec![
                    (2.0, 0.0),
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (-2.0, 0.0),
                    (0.0, 0.0),
                    (0.0, 0.0),
                    (2.0, 0.0),
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (-2.0, 0.0),
                    (0.0, 0.0),
                    (0.0, 0.0),
                ],
                // E2g
                vec![
                    (2.0, 0.0),
                    (-1.0, 0.0),
                    (-1.0, 0.0),
                    (2.0, 0.0),
                    (0.0, 0.0),
                    (0.0, 0.0),
                    (2.0, 0.0),
                    (-1.0, 0.0),
                    (-1.0, 0.0),
                    (2.0, 0.0),
                    (0.0, 0.0),
                    (0.0, 0.0),
                ],
                // A1u
                vec![
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (-1.0, 0.0),
                    (-1.0, 0.0),
                    (-1.0, 0.0),
                    (-1.0, 0.0),
                    (-1.0, 0.0),
                ],
                // A2u
                vec![
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (-1.0, 0.0),
                    (-1.0, 0.0),
                    (-1.0, 0.0),
                    (-1.0, 0.0),
                    (-1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                ],
                // B1u
                vec![
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (-1.0, 0.0),
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (1.0, 0.0),
                ],
                // B2u
                vec![
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (-1.0, 0.0),
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (-1.0, 0.0),
                ],
                // E1u
                vec![
                    (2.0, 0.0),
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (-2.0, 0.0),
                    (0.0, 0.0),
                    (0.0, 0.0),
                    (-2.0, 0.0),
                    (-1.0, 0.0),
                    (1.0, 0.0),
                    (2.0, 0.0),
                    (0.0, 0.0),
                    (0.0, 0.0),
                ],
                // E2u
                vec![
                    (2.0, 0.0),
                    (-1.0, 0.0),
                    (-1.0, 0.0),
                    (2.0, 0.0),
                    (0.0, 0.0),
                    (0.0, 0.0),
                    (-2.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (-2.0, 0.0),
                    (0.0, 0.0),
                    (0.0, 0.0),
                ],
            ],
        }
    }

    /// T: Tetrahedral (12x4)
    fn t() -> Self {
        Self {
            point_group: PointGroup::T,
            irreps: vec![
                IrreducibleRepresentation {
                    label: "A".to_string(),
                    dimension: 1,
                },
                IrreducibleRepresentation {
                    label: "E".to_string(),
                    dimension: 2,
                },
                IrreducibleRepresentation {
                    label: "T".to_string(),
                    dimension: 3,
                },
            ],
            classes: vec![
                ConjugacyClass {
                    name: "E".to_string(),
                    count: 1,
                },
                ConjugacyClass {
                    name: "C3".to_string(),
                    count: 8,
                },
                ConjugacyClass {
                    name: "C2".to_string(),
                    count: 3,
                },
            ],
            characters: vec![
                vec![(1.0, 0.0), (1.0, 0.0), (1.0, 0.0)],
                vec![(2.0, 0.0), (-1.0, 0.0), (2.0, 0.0)],
                vec![(3.0, 0.0), (0.0, 0.0), (-1.0, 0.0)],
            ],
        }
    }

    /// Td: Tetrahedral with mirrors (24x5)
    fn td() -> Self {
        Self {
            point_group: PointGroup::Td,
            irreps: vec![
                IrreducibleRepresentation {
                    label: "A1".to_string(),
                    dimension: 1,
                },
                IrreducibleRepresentation {
                    label: "A2".to_string(),
                    dimension: 1,
                },
                IrreducibleRepresentation {
                    label: "E".to_string(),
                    dimension: 2,
                },
                IrreducibleRepresentation {
                    label: "T1".to_string(),
                    dimension: 3,
                },
                IrreducibleRepresentation {
                    label: "T2".to_string(),
                    dimension: 3,
                },
            ],
            classes: vec![
                ConjugacyClass {
                    name: "E".to_string(),
                    count: 1,
                },
                ConjugacyClass {
                    name: "C3".to_string(),
                    count: 8,
                },
                ConjugacyClass {
                    name: "C2".to_string(),
                    count: 3,
                },
                ConjugacyClass {
                    name: "sigma_d".to_string(),
                    count: 6,
                },
                ConjugacyClass {
                    name: "S4".to_string(),
                    count: 6,
                },
            ],
            characters: vec![
                vec![(1.0, 0.0), (1.0, 0.0), (1.0, 0.0), (1.0, 0.0), (1.0, 0.0)],
                vec![(1.0, 0.0), (1.0, 0.0), (1.0, 0.0), (-1.0, 0.0), (-1.0, 0.0)],
                vec![(2.0, 0.0), (-1.0, 0.0), (2.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
                vec![(3.0, 0.0), (0.0, 0.0), (-1.0, 0.0), (-1.0, 0.0), (1.0, 0.0)],
                vec![(3.0, 0.0), (0.0, 0.0), (-1.0, 0.0), (1.0, 0.0), (-1.0, 0.0)],
            ],
        }
    }

    /// Oh: Octahedral with mirrors (full cubic symmetry) (48x10)
    fn oh() -> Self {
        Self {
            point_group: PointGroup::Oh,
            irreps: vec![
                IrreducibleRepresentation {
                    label: "A1g".to_string(),
                    dimension: 1,
                },
                IrreducibleRepresentation {
                    label: "A2g".to_string(),
                    dimension: 1,
                },
                IrreducibleRepresentation {
                    label: "Eg".to_string(),
                    dimension: 2,
                },
                IrreducibleRepresentation {
                    label: "T1g".to_string(),
                    dimension: 3,
                },
                IrreducibleRepresentation {
                    label: "T2g".to_string(),
                    dimension: 3,
                },
                IrreducibleRepresentation {
                    label: "A1u".to_string(),
                    dimension: 1,
                },
                IrreducibleRepresentation {
                    label: "A2u".to_string(),
                    dimension: 1,
                },
                IrreducibleRepresentation {
                    label: "Eu".to_string(),
                    dimension: 2,
                },
                IrreducibleRepresentation {
                    label: "T1u".to_string(),
                    dimension: 3,
                },
                IrreducibleRepresentation {
                    label: "T2u".to_string(),
                    dimension: 3,
                },
            ],
            classes: vec![
                ConjugacyClass {
                    name: "E".to_string(),
                    count: 1,
                },
                ConjugacyClass {
                    name: "C3".to_string(),
                    count: 8,
                },
                ConjugacyClass {
                    name: "C2".to_string(),
                    count: 3,
                },
                ConjugacyClass {
                    name: "C4".to_string(),
                    count: 6,
                },
                ConjugacyClass {
                    name: "C2'".to_string(),
                    count: 6,
                },
                ConjugacyClass {
                    name: "i".to_string(),
                    count: 1,
                },
                ConjugacyClass {
                    name: "S6".to_string(),
                    count: 8,
                },
                ConjugacyClass {
                    name: "sigma_h".to_string(),
                    count: 3,
                },
                ConjugacyClass {
                    name: "S4".to_string(),
                    count: 6,
                },
                ConjugacyClass {
                    name: "sigma_d".to_string(),
                    count: 6,
                },
            ],
            characters: vec![
                vec![
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                ],
                vec![
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (-1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (-1.0, 0.0),
                ],
                vec![
                    (2.0, 0.0),
                    (-1.0, 0.0),
                    (2.0, 0.0),
                    (0.0, 0.0),
                    (0.0, 0.0),
                    (2.0, 0.0),
                    (-1.0, 0.0),
                    (2.0, 0.0),
                    (0.0, 0.0),
                    (0.0, 0.0),
                ],
                vec![
                    (3.0, 0.0),
                    (0.0, 0.0),
                    (-1.0, 0.0),
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (3.0, 0.0),
                    (0.0, 0.0),
                    (-1.0, 0.0),
                    (-1.0, 0.0),
                    (1.0, 0.0),
                ],
                vec![
                    (3.0, 0.0),
                    (0.0, 0.0),
                    (-1.0, 0.0),
                    (-1.0, 0.0),
                    (1.0, 0.0),
                    (3.0, 0.0),
                    (0.0, 0.0),
                    (-1.0, 0.0),
                    (1.0, 0.0),
                    (-1.0, 0.0),
                ],
                vec![
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (-1.0, 0.0),
                    (-1.0, 0.0),
                    (-1.0, 0.0),
                    (-1.0, 0.0),
                ],
                vec![
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (-1.0, 0.0),
                    (-1.0, 0.0),
                    (-1.0, 0.0),
                    (-1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                ],
                vec![
                    (2.0, 0.0),
                    (-1.0, 0.0),
                    (2.0, 0.0),
                    (0.0, 0.0),
                    (0.0, 0.0),
                    (-2.0, 0.0),
                    (1.0, 0.0),
                    (-2.0, 0.0),
                    (0.0, 0.0),
                    (0.0, 0.0),
                ],
                vec![
                    (3.0, 0.0),
                    (0.0, 0.0),
                    (-1.0, 0.0),
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (-3.0, 0.0),
                    (0.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (-1.0, 0.0),
                ],
                vec![
                    (3.0, 0.0),
                    (0.0, 0.0),
                    (-1.0, 0.0),
                    (-1.0, 0.0),
                    (1.0, 0.0),
                    (-3.0, 0.0),
                    (0.0, 0.0),
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (1.0, 0.0),
                ],
            ],
        }
    }
}

impl fmt::Display for CharacterTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Character Table for {}", self.point_group)?;
        writeln!(f, "{:-^80}", " ")?;

        // Header row with class names
        write!(f, "{:12}", "Irrep")?;
        for class in &self.classes {
            write!(f, "{:12}", class.name)?;
        }
        writeln!(f)?;
        writeln!(f, "{:-^80}", " ")?;

        // Data rows
        for (i, irrep) in self.irreps.iter().enumerate() {
            write!(f, "{:12}", irrep.label)?;
            for j in 0..self.classes.len() {
                let (re, im) = self.characters[i][j];
                if im.abs() < 1e-10 {
                    write!(f, "{:12.1}", re)?;
                } else if re.abs() < 1e-10 {
                    write!(f, "{:11.1}i", im)?;
                } else {
                    write!(f, "{:8.1}+{:3.1}i", re, im)?;
                }
            }
            writeln!(f)?;
        }
        writeln!(f)?;
        Ok(())
    }
}

#[cfg(test)]
mod codegen_parity_tests {
    //! Regression tests pinning the Phase 7 #127 codegen-vs-inline parity
    //! for the 5 migrated abelian point groups. Each test reads the
    //! codegen path AND the inline path and asserts they agree on every
    //! field. If they ever drift, the test catches it immediately.

    use super::*;

    fn assert_tables_equal(a: &CharacterTable, b: &CharacterTable) {
        assert_eq!(a.point_group, b.point_group);
        assert_eq!(a.classes.len(), b.classes.len(), "class count");
        for (ca, cb) in a.classes.iter().zip(b.classes.iter()) {
            assert_eq!(ca.name, cb.name);
            assert_eq!(ca.count, cb.count);
        }
        assert_eq!(a.irreps.len(), b.irreps.len(), "irrep count");
        for (ia, ib) in a.irreps.iter().zip(b.irreps.iter()) {
            assert_eq!(ia.label, ib.label);
            assert_eq!(ia.dimension, ib.dimension);
        }
        assert_eq!(a.characters.len(), b.characters.len(), "matrix rows");
        for (ra, rb) in a.characters.iter().zip(b.characters.iter()) {
            assert_eq!(ra.len(), rb.len(), "matrix row width");
            for ((re_a, im_a), (re_b, im_b)) in ra.iter().zip(rb.iter()) {
                assert!(
                    (re_a - re_b).abs() < 1e-12,
                    "re drift: {} vs {}",
                    re_a,
                    re_b
                );
                assert!(
                    (im_a - im_b).abs() < 1e-12,
                    "im drift: {} vs {}",
                    im_a,
                    im_b
                );
            }
        }
    }

    #[test]
    fn c1_codegen_matches_inline() {
        let codegen = from_codegen_table(PointGroup::C1).expect("C1 in registry");
        let inline = CharacterTable::c1();
        assert_tables_equal(&codegen, &inline);
    }

    #[test]
    fn ci_codegen_matches_inline() {
        let codegen = from_codegen_table(PointGroup::Ci).expect("Ci in registry");
        let inline = CharacterTable::ci();
        assert_tables_equal(&codegen, &inline);
    }

    #[test]
    fn c2_codegen_matches_inline() {
        let codegen = from_codegen_table(PointGroup::C2).expect("C2 in registry");
        let inline = CharacterTable::c2();
        assert_tables_equal(&codegen, &inline);
    }

    #[test]
    fn cs_codegen_matches_inline() {
        let codegen = from_codegen_table(PointGroup::Cs).expect("Cs in registry");
        let inline = CharacterTable::cs();
        assert_tables_equal(&codegen, &inline);
    }

    #[test]
    fn c2h_codegen_matches_inline() {
        let codegen = from_codegen_table(PointGroup::C2h).expect("C2h in registry");
        let inline = CharacterTable::c2h();
        assert_tables_equal(&codegen, &inline);
    }

    #[test]
    fn all_thirty_two_crystallographic_point_groups_have_codegen_tables() {
        // After #127 Phase 7 completion, every one of the 32 crystallographic
        // point groups has a TOML entry. This is the regression test that
        // catches accidental removal of any group from
        // character_tables.toml.
        use super::PointGroup as P;
        for pg in [
            P::C1,
            P::Ci,
            P::C2,
            P::Cs,
            P::C2h,
            P::D2,
            P::C2v,
            P::D2h,
            P::C4,
            P::S4,
            P::C4h,
            P::D4,
            P::C4v,
            P::D2d,
            P::D4h,
            P::C3,
            P::C3i,
            P::C3v,
            P::D3,
            P::D3d,
            P::C6,
            P::C3h,
            P::C6h,
            P::D6,
            P::C6v,
            P::D3h,
            P::D6h,
            P::T,
            P::Td,
            P::Th,
            P::O,
            P::Oh,
        ] {
            assert!(
                from_codegen_table(pg).is_some(),
                "{:?} missing from CHARACTER_TABLE_REGISTRY -- regression vs #127 Phase 7 completion",
                pg
            );
        }
    }

    #[test]
    fn schur_orthogonality_for_real_form_tables() {
        // For the dim=2 real-form irreps used in cyclic groups,
        // sum_k n_k * |chi_E(C_k)|^2 should equal 2 * |G|.
        // We verify this on D2d as a representative non-abelian group
        // with an E irrep.
        let t = from_codegen_table(PointGroup::D2d).expect("D2d in registry");
        let g_order: f64 = t.classes.iter().map(|c| c.count as f64).sum();
        // E is the 5th irrep (index 4): chi at E,2S4,C2,2C2',2sigma_d = 2,0,-2,0,0
        let row_e = &t.characters[4];
        let sum_sq: f64 = t
            .classes
            .iter()
            .zip(row_e.iter())
            .map(|(c, &(re, im))| (c.count as f64) * (re * re + im * im))
            .sum();
        // For a 2D real irrep, expected = 2 * |G|.
        // sum_sq = 1*4 + 2*0 + 1*4 + 2*0 + 2*0 = 8, |G| = 1+2+1+2+2 = 8, so 8 = 2*4? No, 8 = |G|.
        // Real-form 2D irreps satisfy sum_sq = |G| (one dimension's worth of
        // orthogonality per the two paired complex irreps). This is the
        // convention used by the codebase.
        assert!(
            (sum_sq - g_order).abs() < 1e-9,
            "D2d E-irrep real-form orthogonality: sum_sq={} should equal |G|={}",
            sum_sq,
            g_order
        );
    }
}
