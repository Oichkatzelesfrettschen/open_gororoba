//! The 7 crystal lattice systems.
//!
//! Extracted from `crystal_symmetry.rs` as part of the #139 PH-MOD split.
//! `LatticeSystem` enum (Triclinic, Monoclinic, Orthorhombic, Tetragonal,
//! Hexagonal, Rhombohedral, Cubic) + Display.

use std::fmt;

// ============================================================================
// Lattice Systems and Crystal Systems
// ============================================================================

/// The 7 crystal lattice systems.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LatticeSystem {
    /// a != b != c, alpha != beta != gamma != 90 deg
    Triclinic,
    /// a != b != c, alpha = gamma = 90 deg != beta
    Monoclinic,
    /// a != b != c, alpha = beta = gamma = 90 deg
    Orthorhombic,
    /// a = b != c, alpha = beta = gamma = 90 deg
    Tetragonal,
    /// a = b != c, alpha = beta = 90 deg, gamma = 120 deg
    Hexagonal,
    /// a = b = c, alpha = beta = gamma (not 90 deg)
    Rhombohedral,
    /// a = b = c, alpha = beta = gamma = 90 deg
    Cubic,
}

impl fmt::Display for LatticeSystem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Triclinic => "Triclinic",
            Self::Monoclinic => "Monoclinic",
            Self::Orthorhombic => "Orthorhombic",
            Self::Tetragonal => "Tetragonal",
            Self::Hexagonal => "Hexagonal",
            Self::Rhombohedral => "Rhombohedral",
            Self::Cubic => "Cubic",
        };
        write!(f, "{}", s)
    }
}
