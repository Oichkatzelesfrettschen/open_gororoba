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

impl CharacterTable {
    /// Get character table for a given point group.
    /// Returns Some(table) for supported groups, None otherwise.
    pub fn for_point_group(pg: PointGroup) -> Option<Self> {
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
