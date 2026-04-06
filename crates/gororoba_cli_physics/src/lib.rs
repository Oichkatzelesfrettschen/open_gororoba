//! Physics domain CLI binaries for gororoba.
//!
//! This crate contains 58 binaries covering GR, cosmology, optics, materials,
//! quantum mechanics, spectral analysis, fluids, HEP/QGP, and vacuum physics.

pub mod anomaly_residual;
pub mod ephemeris_loader;
pub mod flyby;
pub mod heliosphere_boundary;
pub mod heliosphere_eval;
pub mod lbm_dispatch;
pub mod nonlocal_report;
pub mod voyager_arrow;
pub mod voyager_encounter;

#[cfg(test)]
#[test]
fn takens_descriptor_sedenion_lane_matches_scalar_reference() {
    crate::heliosphere_eval::assert_takens_descriptor_sedenion_lane_matches_scalar_reference();
}
