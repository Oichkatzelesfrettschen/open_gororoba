//! E8 x E8 Heterotic String Theory Connection.
//!
//! Models the gauge group symmetry breaking and the role of division algebras
//! in the Standard Model embedding via E8.

use crate::lie::e8_lattice::DivisionAlgebra;

/// Gauge group for heterotic string theory.
pub enum HeteroticGaugeGroup {
    E8xE8,
    SO32,
}

/// Representation of the E8 symmetry breaking to SU(3) x E6.
pub struct E8SymmetryBreaking {
    pub parent_group: &'static str,
    pub subgroups: Vec<&'static str>,
}

impl E8SymmetryBreaking {
    pub fn standard_model_embedding() -> Self {
        Self {
            parent_group: "E8",
            subgroups: vec!["SU(3)", "E6"],
        }
    }

    /// Octonionic projective plane OP^2 has isometry group E6.
    pub fn division_algebra_isometry(algebra: DivisionAlgebra) -> Option<&'static str> {
        match algebra {
            DivisionAlgebra::O => Some("E6"),
            _ => None,
        }
    }
}

/// Anomaly cancellation requirement for heterotic strings.
/// dim(G) must be 496.
pub fn verify_anomaly_cancellation(group: HeteroticGaugeGroup) -> bool {
    match group {
        HeteroticGaugeGroup::E8xE8 => 248 * 2 == 496,
        HeteroticGaugeGroup::SO32 => (32 * 31) / 2 == 496,
    }
}
