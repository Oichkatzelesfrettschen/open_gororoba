//! Heterotic-string facade for the two anomaly-free 10D gauge choices.
//!
//! Heterotic string theories admit exactly two consistent gauge groups in 10D
//! (Green-Schwarz anomaly cancellation):
//! - `E_8 x E_8` (heterotic-E8), with `dim = 248 + 248 = 496`.
//! - `Spin(32)/Z_2` (heterotic-O), with `dim = 32 * 31 / 2 = 496`.
//!
//! Both groups have dimension 496 -- the unique anomaly-cancellation requirement.
//!
//! # Standard Model embedding
//!
//! The canonical chain `E_8 -> SU(3) x E_6` (one of the two `E_8` factors
//! breaks to a visible-sector `E_6` GUT plus a colour group) is what makes
//! heterotic strings a candidate Theory of Everything. The hidden `E_8` factor
//! couples only gravitationally and is the natural home of dark-sector content.
//!
//! # Reference
//! - Green, Schwarz, Witten, *Superstring Theory* vol. 2 (1987), Ch. 6.
//! - Polchinski, *String Theory* vol. 2 (1998), Ch. 11.

use super::magic_square::DivisionAlgebra;

// ============================================================================
// Types
// ============================================================================

/// One of the two anomaly-free heterotic gauge groups.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeteroticGaugeGroup {
    /// `E_8 x E_8` (heterotic-E8 string).
    E8xE8,
    /// `Spin(32)/Z_2` (heterotic-O / `SO(32)` string).
    SO32,
}

impl HeteroticGaugeGroup {
    /// Group dimension (must be 496 for anomaly cancellation).
    pub fn dim(&self) -> usize {
        match self {
            Self::E8xE8 => 248 + 248,
            Self::SO32 => 32 * 31 / 2,
        }
    }
}

/// Canonical breaking pattern from `E_8` to standard-model precursors.
#[derive(Debug, Clone)]
pub struct E8SymmetryBreaking {
    /// Parent group label.
    pub parent_group: &'static str,
    /// Direct subgroups produced by the breaking.
    pub subgroups: Vec<&'static str>,
}

impl E8SymmetryBreaking {
    /// `E_8 -> SU(3) x E_6`: standard-model gauge plus visible-sector GUT.
    pub fn standard_model_embedding() -> Self {
        Self {
            parent_group: "E8",
            subgroups: vec!["SU(3)", "E6"],
        }
    }

    /// Isometry group of the projective plane `K P^2` for each division algebra.
    ///
    /// `OP^2` (octonionic projective plane) has isometry group `F_4`, but for
    /// the *complexified* / supersymmetric version associated with `E_6` we
    /// return `E_6` for octonions; the lower-dimensional planes are unrelated
    /// to exceptional groups.
    pub fn division_algebra_isometry(algebra: DivisionAlgebra) -> Option<&'static str> {
        match algebra {
            DivisionAlgebra::O => Some("E6"),
            _ => None,
        }
    }
}

// ============================================================================
// Operations
// ============================================================================

/// Verify the Green-Schwarz anomaly-cancellation requirement: `dim(G) == 496`.
pub fn verify_anomaly_cancellation(group: HeteroticGaugeGroup) -> bool {
    group.dim() == 496
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_gauge_groups_have_dim_496() {
        assert_eq!(HeteroticGaugeGroup::E8xE8.dim(), 496);
        assert_eq!(HeteroticGaugeGroup::SO32.dim(), 496);
    }

    #[test]
    fn anomaly_cancellation_holds_for_both_groups() {
        assert!(verify_anomaly_cancellation(HeteroticGaugeGroup::E8xE8));
        assert!(verify_anomaly_cancellation(HeteroticGaugeGroup::SO32));
    }

    #[test]
    fn standard_model_embedding_uses_su3_e6_chain() {
        let e = E8SymmetryBreaking::standard_model_embedding();
        assert_eq!(e.parent_group, "E8");
        assert!(e.subgroups.contains(&"SU(3)"));
        assert!(e.subgroups.contains(&"E6"));
    }

    #[test]
    fn only_octonions_yield_an_exceptional_isometry() {
        assert_eq!(
            E8SymmetryBreaking::division_algebra_isometry(DivisionAlgebra::O),
            Some("E6")
        );
        assert_eq!(
            E8SymmetryBreaking::division_algebra_isometry(DivisionAlgebra::H),
            None
        );
        assert_eq!(
            E8SymmetryBreaking::division_algebra_isometry(DivisionAlgebra::C),
            None
        );
        assert_eq!(
            E8SymmetryBreaking::division_algebra_isometry(DivisionAlgebra::R),
            None
        );
    }
}
