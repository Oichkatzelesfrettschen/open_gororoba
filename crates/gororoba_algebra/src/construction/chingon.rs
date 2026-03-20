// Canonical 64D type lives in cd_tower; re-export for backward compat.
pub use crate::construction::cd_tower::Chingon;

// AVT types moved to algebra_analysis::avt (Phase 3).
// Re-exported here so all existing import paths remain valid.
pub use algebra_analysis::avt::{
    AlternativityViolationTensor, PackedAvt, SampledAvt, index_bits_for_dim,
};

#[cfg(test)]
#[path = "chingon_tests.rs"]
mod tests;
