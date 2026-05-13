//! Space groups (one of 230) with Hermann-Mauguin + Schoenflies symbols.
//!
//! Extracted from `crystal_symmetry.rs` as part of the #139 PH-MOD split.
//! `SpaceGroup` struct combines point-group symmetry with translation,
//! and the CSV-name conversion helpers used by the build.rs codegen lane
//! to populate the 230-entry table from the ITA reference data.

use std::fmt;

use super::{LatticeSystem, PointGroup};

// ============================================================================
// Space Groups (230 Total)
// ============================================================================

/// A crystallographic space group (one of 230).
///
/// Space groups combine point group symmetry with translation.
#[derive(Debug, Clone)]
pub struct SpaceGroup {
    /// Space group number (1-230)
    pub number: u16,
    /// Hermann-Mauguin symbol (e.g., "P21/c")
    pub hm_symbol: &'static str,
    /// Schoenflies symbol (e.g., "C2h^4")
    pub schoenflies_symbol: &'static str,
    /// Point group symmetry
    pub point_group: PointGroup,
    /// Lattice system
    pub lattice_system: LatticeSystem,
    /// Bravais lattice centering: P (primitive), C (base-centered), F (face-centered), I (body-centered), R (rhombohedral)
    pub bravais_centering: char,
}

impl SpaceGroup {
    /// Get space group by number (1-230).
    ///
    /// Uses the full ITA table generated from `materials_data/data/space_groups.csv` at
    /// compile time. All 230 groups are covered; returns `None` only for n outside 1..=230.
    pub fn from_number(n: u16) -> Option<Self> {
        if !(1..=230).contains(&n) {
            return None;
        }
        let row = materials_data::SPACE_GROUP_TABLE[(n as usize) - 1];
        Some(SpaceGroup {
            number: row.0,
            hm_symbol: row.1,
            schoenflies_symbol: row.2,
            point_group: point_group_from_str(row.3),
            lattice_system: lattice_system_from_str(row.4),
            bravais_centering: row.5,
        })
    }

    /// All space groups for a given point group (linear scan of the full ITA table).
    pub fn for_point_group(pg: PointGroup) -> Vec<SpaceGroup> {
        materials_data::SPACE_GROUP_TABLE
            .iter()
            .filter(|row| point_group_from_str(row.3) == pg)
            .map(|row| SpaceGroup {
                number: row.0,
                hm_symbol: row.1,
                schoenflies_symbol: row.2,
                point_group: pg,
                lattice_system: lattice_system_from_str(row.4),
                bravais_centering: row.5,
            })
            .collect()
    }
}

/// Convert a PointGroup CSV variant name to the enum variant.
fn point_group_from_str(s: &str) -> PointGroup {
    match s {
        "C1" => PointGroup::C1,
        "Ci" => PointGroup::Ci,
        "C2" => PointGroup::C2,
        "Cs" => PointGroup::Cs,
        "C2h" => PointGroup::C2h,
        "D2" => PointGroup::D2,
        "C2v" => PointGroup::C2v,
        "D2h" => PointGroup::D2h,
        "C4" => PointGroup::C4,
        "S4" => PointGroup::S4,
        "C4h" => PointGroup::C4h,
        "D4" => PointGroup::D4,
        "C4v" => PointGroup::C4v,
        "D2d" => PointGroup::D2d,
        "D4h" => PointGroup::D4h,
        "C3" => PointGroup::C3,
        "C3i" => PointGroup::C3i,
        "C3v" => PointGroup::C3v,
        "D3" => PointGroup::D3,
        "D3d" => PointGroup::D3d,
        "C6" => PointGroup::C6,
        "C3h" => PointGroup::C3h,
        "C6h" => PointGroup::C6h,
        "D6" => PointGroup::D6,
        "C6v" => PointGroup::C6v,
        "D3h" => PointGroup::D3h,
        "D6h" => PointGroup::D6h,
        "T" => PointGroup::T,
        "Td" => PointGroup::Td,
        "Th" => PointGroup::Th,
        "O" => PointGroup::O,
        "Oh" => PointGroup::Oh,
        other => panic!("unknown PointGroup in space_groups.csv: {other:?}"),
    }
}

/// Convert a LatticeSystem CSV variant name to the enum variant.
fn lattice_system_from_str(s: &str) -> LatticeSystem {
    match s {
        "Triclinic" => LatticeSystem::Triclinic,
        "Monoclinic" => LatticeSystem::Monoclinic,
        "Orthorhombic" => LatticeSystem::Orthorhombic,
        "Tetragonal" => LatticeSystem::Tetragonal,
        "Hexagonal" => LatticeSystem::Hexagonal,
        "Rhombohedral" => LatticeSystem::Rhombohedral,
        "Cubic" => LatticeSystem::Cubic,
        other => panic!("unknown LatticeSystem in space_groups.csv: {other:?}"),
    }
}

impl fmt::Display for SpaceGroup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Sg#{} {} ({}) [{}]",
            self.number, self.hm_symbol, self.schoenflies_symbol, self.point_group
        )
    }
}
