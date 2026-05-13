//! The 32 crystallographic point groups (Schoenflies notation) + their
//! order and identification helpers.
//!
//! Extracted from `crystal_symmetry.rs` as part of the #139 PH-MOD split.
//! Includes the canonical `PointGroup` enum (32 variants from C1 to Oh),
//! `order()` for the group's number of symmetry operations, and Display
//! using Schoenflies symbols.

use std::fmt;

use super::LatticeSystem;

// ============================================================================
// Point Groups (32 Crystallographic Point Groups)
// ============================================================================

/// The 32 crystallographic point groups (Schoenflies notation).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PointGroup {
    // Triclinic
    C1, // Identity only
    Ci, // Inversion

    // Monoclinic
    C2,  // 2-fold rotation
    Cs,  // Mirror plane
    C2h, // 2-fold rotation + mirror

    // Orthorhombic
    D2,  // Three perpendicular 2-fold rotations (mmm in tetragonal)
    C2v, // 2-fold rotation + two mirror planes
    D2h, // Three 2-fold + three mirrors (full mmm)

    // Tetragonal
    C4,  // 4-fold rotation
    S4,  // 4-fold improper (inversion axis of order 4)
    C4h, // 4-fold + mirror plane
    D4,  // 4-fold + four 2-folds
    C4v, // 4-fold + two mirrors
    D2d, // Four 2-fold axes + dihedral (dihedral group)
    D4h, // 4-fold + four 2-folds + mirrors

    // Trigonal (Rhombohedral)
    C3,  // 3-fold rotation
    C3i, // 3-fold + inversion
    C3v, // 3-fold + three mirrors
    D3,  // 3-fold + three 2-folds
    D3d, // 3-fold + mirrors + inversion

    // Hexagonal
    C6,  // 6-fold rotation
    C3h, // 3-fold + mirror
    C6h, // 6-fold + mirror
    D6,  // 6-fold + six 2-folds
    C6v, // 6-fold + six mirrors
    D3h, // 3-fold + three 2-folds + mirrors (dihedral hexagonal)
    D6h, // 6-fold + mirrors (full hexagonal symmetry)

    // Cubic
    T,  // Tetrahedral (4 three-fold axes)
    Td, // Tetrahedral + mirrors
    Th, // Tetrahedral + inversion
    O,  // Octahedral (3 four-fold axes)
    Oh, // Octahedral + mirrors (full cubic symmetry)
}

impl PointGroup {
    /// Order of the point group (number of symmetry operations).
    pub fn order(&self) -> usize {
        match self {
            Self::C1 => 1,
            Self::Ci | Self::Cs | Self::C2 => 2,
            Self::C2h | Self::C2v => 4,
            Self::D2 => 4,
            Self::D2h => 8,
            Self::C3 => 3,
            Self::C3v | Self::D3 => 6,
            Self::C3i | Self::D3d => 6,
            Self::C4 | Self::S4 => 4,
            Self::C4h | Self::C4v => 8,
            Self::D2d | Self::D4 => 8,
            Self::D4h => 16,
            Self::C6 | Self::C3h => 6,
            Self::C6h => 12,
            Self::C6v | Self::D3h => 12,
            Self::D6 => 12,
            Self::D6h => 24,
            Self::T => 12,
            Self::Td | Self::Th => 24,
            Self::O => 24,
            Self::Oh => 48,
        }
    }

    /// Lattice system compatibility.
    pub fn lattice_system(&self) -> LatticeSystem {
        match self {
            Self::C1 | Self::Ci => LatticeSystem::Triclinic,
            Self::C2 | Self::Cs | Self::C2h => LatticeSystem::Monoclinic,
            Self::D2 | Self::C2v | Self::D2h => LatticeSystem::Orthorhombic,
            Self::C4 | Self::S4 | Self::C4h | Self::D4 | Self::C4v | Self::D2d | Self::D4h => {
                LatticeSystem::Tetragonal
            }
            Self::C3 | Self::C3i | Self::C3v | Self::D3 | Self::D3d => LatticeSystem::Rhombohedral,
            Self::C6 | Self::C3h | Self::C6h | Self::D6 | Self::C6v | Self::D3h | Self::D6h => {
                LatticeSystem::Hexagonal
            }
            Self::T | Self::Td | Self::Th | Self::O | Self::Oh => LatticeSystem::Cubic,
        }
    }

    /// All 32 point groups.
    pub fn all() -> &'static [PointGroup; 32] {
        &[
            Self::C1,
            Self::Ci,
            Self::C2,
            Self::Cs,
            Self::C2h,
            Self::D2,
            Self::C2v,
            Self::D2h,
            Self::C3,
            Self::C3i,
            Self::C3v,
            Self::D3,
            Self::D3d,
            Self::C4,
            Self::S4,
            Self::C4h,
            Self::D4,
            Self::C4v,
            Self::D2d,
            Self::D4h,
            Self::C6,
            Self::C3h,
            Self::C6h,
            Self::D6,
            Self::C6v,
            Self::D3h,
            Self::D6h,
            Self::T,
            Self::Td,
            Self::Th,
            Self::O,
            Self::Oh,
        ]
    }
}

impl fmt::Display for PointGroup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::C1 => "C1",
            Self::Ci => "Ci",
            Self::C2 => "C2",
            Self::Cs => "Cs",
            Self::C2h => "C2h",
            Self::D2 => "D2",
            Self::C2v => "C2v",
            Self::D2h => "D2h",
            Self::C3 => "C3",
            Self::C3i => "C3i",
            Self::C3v => "C3v",
            Self::D3 => "D3",
            Self::D3d => "D3d",
            Self::C4 => "C4",
            Self::S4 => "S4",
            Self::C4h => "C4h",
            Self::D4 => "D4",
            Self::C4v => "C4v",
            Self::D2d => "D2d",
            Self::D4h => "D4h",
            Self::C6 => "C6",
            Self::C3h => "C3h",
            Self::C6h => "C6h",
            Self::D6 => "D6",
            Self::C6v => "C6v",
            Self::D3h => "D3h",
            Self::D6h => "D6h",
            Self::T => "T",
            Self::Td => "Td",
            Self::Th => "Th",
            Self::O => "O",
            Self::Oh => "Oh",
        };
        write!(f, "{}", s)
    }
}
